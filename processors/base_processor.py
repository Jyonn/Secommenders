import json
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from utils.artifact import ArtifactStore
from utils.pipeline import ensure_formatted
from utils.stable_random import stable_shuffle


class Processor:
    VER = 'v2.6'

    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000
    VALID_RATIO = 0.2

    def __init__(self, dataset: str, num_test=None, num_finetune=None):
        self.dataset = dataset.lower()

        self.num_test = self.NUM_TEST if num_test is None else num_test
        self.num_finetune = self.NUM_FINETUNE if num_finetune is None else num_finetune

        self.store = ArtifactStore(self.get_name())
        self.formatted_dir = self.store.formatted_dir()
        self.store_dir = str(self.store.processed_dir())
        self._loaded = False

        self.IID_COL: Optional[str] = None
        self.UID_COL: Optional[str] = None
        self.HIS_COL: Optional[str] = None
        self.REQUIRE_STRINGIFY: Optional[bool] = None
        self._default_attrs = []
        self.provides_test_set = False
        self.multi_item_col: Optional[str] = None
        self.use_all_users_in_processor = False
        self.split_ratio = 0.9
        self.remaining_users_as_valid = False
        self.user_order_seed: Optional[str] = None

        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

        self.formatted_test_set: Optional[pd.DataFrame] = None
        self.test_set: Optional[pd.DataFrame] = None
        self.valid_set: Optional[pd.DataFrame] = None
        self.finetune_set: Optional[pd.DataFrame] = None

    def get_name(self):
        return self.dataset

    @property
    def default_attrs(self):
        return self._default_attrs

    def _paths(self):
        base_dir = Path(self.store_dir)
        return {
            'items': base_dir / 'items.parquet',
            'valid': base_dir / 'valid.parquet',
            'test': base_dir / 'test.parquet',
            'finetune': base_dir / 'finetune.parquet',
            'user_order': base_dir / 'user_order.txt',
            'meta': base_dir / 'meta.json',
            'stats': base_dir / 'stats.json',
        }

    def _formatted_paths(self):
        return {
            'items': self.formatted_dir / 'items.parquet',
            'users': self.formatted_dir / 'users.parquet',
            'test_users': self.formatted_dir / 'test_users.parquet',
            'meta': self.formatted_dir / 'meta.json',
            'stats': self.formatted_dir / 'stats.json',
        }

    def _stringify(self, df: pd.DataFrame):
        if not self.REQUIRE_STRINGIFY:
            return df
        if self.IID_COL in df.columns:
            df[self.IID_COL] = df[self.IID_COL].astype(str)
        if self.UID_COL in df.columns:
            df[self.UID_COL] = df[self.UID_COL].astype(str)
        return df

    def _load_meta(self, path: Path):
        return json.loads(path.read_text())

    def _apply_meta(self, meta):
        self.IID_COL = meta['item_col']
        self.UID_COL = meta['user_col']
        self.HIS_COL = meta['history_col']
        self.REQUIRE_STRINGIFY = bool(meta.get('require_stringify', False))
        self._default_attrs = list(meta.get('default_attrs', []))
        self.provides_test_set = bool(meta.get('provides_test_set', False))
        self.multi_item_col = meta.get('multi_item_col')
        self.use_all_users_in_processor = bool(meta.get('use_all_users_in_processor', False))
        self.split_ratio = float(meta.get('split_ratio', 0.9))
        self.remaining_users_as_valid = bool(meta.get('remaining_users_as_valid', False))
        self.user_order_seed = str(meta.get('user_order_seed') or f'{self.get_name()}:user-order')

    def _load_formatted_meta(self):
        path = self._formatted_paths()['meta']
        if not path.exists():
            ensure_formatted(self.dataset)
        meta = self._load_meta(path)
        self._apply_meta(meta)
        return meta

    def _load_processed_meta(self):
        path = self._paths()['meta']
        if not path.exists():
            return None
        meta = self._load_meta(path)
        self._apply_meta(meta)
        return meta

    def organize_item(self, iid, item_attrs: list, as_dict=False, item_self=False):
        item = iid if item_self else self.items.iloc[self.item_vocab[iid]]

        if as_dict:
            return {attr: item[attr] or '' for attr in item_attrs}
        if len(item_attrs) == 1:
            return item[item_attrs[0]]
        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])

    @staticmethod
    def _build_slicer(slicer: int):
        def _slicer(x):
            return x[:slicer] if slicer > 0 else x[slicer:]

        return _slicer

    def _iterate(self, dataframe: pd.DataFrame, slicer: Union[int, Callable], **kwargs):
        if isinstance(slicer, int):
            slicer = self._build_slicer(slicer)

        for _, row in dataframe.iterrows():
            uid = row[self.UID_COL]
            history = slicer(row[self.HIS_COL])
            yield uid, history

    def get_source_set(self, source):
        assert source in ['test', 'valid', 'finetune', 'original'], 'source must be test, valid, finetune, or original'
        if source == 'original':
            self._ensure_original_users_loaded()
            return self.users
        return getattr(self, f'{source}_set')

    def generate(self, slicer: Union[int, Callable], source='test', **kwargs):
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        source_set = self.get_source_set(source)
        return self._iterate(source_set, slicer)

    def iterate(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='original')

    def test(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='test')

    def valid(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='valid')

    def finetune(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='finetune')

    def _iterator(self, user_order, users):
        users_by_id = users.set_index(self.UID_COL, drop=False)
        for uid in user_order:
            user = users_by_id.loc[uid]
            if isinstance(user, pd.DataFrame):
                yield user.iloc[0]
                continue
            yield user

    def _split(self, iterator, count):
        users = []
        for user in tqdm(iterator, total=count):
            users.append(user)
            if len(users) >= count:
                break
        return pd.DataFrame(users)

    def _load_user_order(self):
        path = self._paths()['user_order']
        users = self.users[self.UID_COL].unique().tolist()
        users = sorted(users, key=lambda value: str(value))
        user_order = stable_shuffle(users, seed=str(self.user_order_seed or f'{self.get_name()}:user-order'))
        path.write_text(''.join(f'{user}\n' for user in user_order))
        return user_order

    @property
    def test_set_required(self):
        return self.use_all_users_in_processor or self.num_test > 0

    @property
    def valid_set_required(self):
        return self.test_set_required

    @property
    def finetune_set_required(self):
        return self.use_all_users_in_processor or self.num_finetune > 0

    @staticmethod
    def _parquet_file_looks_valid(path: Path):
        try:
            if path.stat().st_size < 8:
                return False
            with path.open('rb') as file:
                head = file.read(4)
                file.seek(-4, 2)
                tail = file.read(4)
            return head == b'PAR1' and tail == b'PAR1'
        except OSError:
            return False

    def _formatted_cache_current(self):
        paths = self._formatted_paths()
        required = [paths['items'], paths['users'], paths['meta']]
        if not all(path.exists() for path in required):
            return False
        try:
            meta = self._load_meta(paths['meta'])
            from utils.data import get_data_dir
            from utils.function import load_formatter

            formatter = load_formatter(self.dataset, data_dir=get_data_dir(self.dataset))
            if formatter.PROVIDES_TEST_SET and not paths['test_users'].exists():
                return False
            return formatter.cache_meta_valid(meta)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return False

    @property
    def processed_valid(self):
        paths = self._paths()
        required = [paths['items'], paths['meta']]
        if self.valid_set_required:
            required.append(paths['valid'])
        if self.test_set_required:
            required.append(paths['test'])
        if self.finetune_set_required:
            required.append(paths['finetune'])
        if not all(path.exists() for path in required):
            return False
        corrupt_parquets = [
            path
            for path in required
            if path.suffix == '.parquet' and not self._parquet_file_looks_valid(path)
        ]
        if corrupt_parquets:
            pnt(
                f'processed {self.get_name()} cache has invalid parquet file(s): '
                f'{", ".join(str(path) for path in corrupt_parquets)}; rebuilding cache'
            )
            return False

        processed_meta = self._load_meta(paths['meta'])
        formatted_meta_path = self._formatted_paths()['meta']
        if not self._formatted_cache_current():
            pnt(
                f'formatted {self.get_name()} cache is stale for current formatter code; '
                'rebuilding formatted and processed artifacts'
            )
            return False
        counts_match = (
            bool(processed_meta.get('use_all_users_in_processor', False))
            or (
                int(processed_meta.get('num_test', -1)) == int(self.num_test)
                and int(processed_meta.get('num_finetune', -1)) == int(self.num_finetune)
            )
        )
        if not formatted_meta_path.exists():
            return (
                processed_meta.get('version') == self.VER
                and counts_match
                and float(processed_meta.get('valid_ratio', -1)) == float(self.VALID_RATIO)
            )
        formatted_meta = self._load_meta(formatted_meta_path)

        return (
            processed_meta.get('version') == self.VER
            and processed_meta.get('formatted_version') == formatted_meta.get('version')
            and counts_match
            and float(processed_meta.get('valid_ratio', -1)) == float(self.VALID_RATIO)
            and str(processed_meta.get('user_order_seed') or '') == str(formatted_meta.get('user_order_seed') or '')
            and float(processed_meta.get('split_ratio', -1.0)) == float(formatted_meta.get('split_ratio', 0.9))
            and bool(processed_meta.get('remaining_users_as_valid', False))
            == bool(formatted_meta.get('remaining_users_as_valid', False))
        )

    def load_formatted(self):
        paths = self._formatted_paths()
        if not self._formatted_cache_current():
            ensure_formatted(self.dataset)
        meta = self._load_formatted_meta()

        self.items = self._stringify(pd.read_parquet(paths['items']))
        self.users = self._stringify(pd.read_parquet(paths['users']))
        if self.provides_test_set:
            if not paths['test_users'].exists():
                ensure_formatted(self.dataset)
            self.formatted_test_set = self._stringify(pd.read_parquet(paths['test_users']))
        if self.REQUIRE_STRINGIFY:
            self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(lambda x: [str(item) for item in x])
            if self.formatted_test_set is not None:
                self.formatted_test_set[self.HIS_COL] = self.formatted_test_set[self.HIS_COL].apply(
                    lambda x: [str(item) for item in x]
                )
                if self.multi_item_col and self.multi_item_col in self.formatted_test_set.columns:
                    self.formatted_test_set[self.multi_item_col] = self.formatted_test_set[self.multi_item_col].apply(
                        lambda x: [str(item) for item in x]
                    )
        return meta

    def _ensure_original_users_loaded(self):
        if self.users is not None:
            return
        self.load_formatted()

    def _collect_public_item_set(self):
        if not self.test_set_required and not self.finetune_set_required:
            return set(self.items[self.IID_COL].unique())

        item_set = set()
        for dataframe in [self.valid_set, self.test_set, self.finetune_set]:
            if dataframe is None:
                continue
            dataframe[self.HIS_COL].apply(lambda x: [item_set.add(i) for i in x])
            if self.multi_item_col and self.multi_item_col in dataframe.columns:
                dataframe[self.multi_item_col].apply(lambda x: [item_set.add(i) for i in x])
        return item_set

    def _build_processed_items(self):
        item_set = self._collect_public_item_set()
        items = self.items[self.items[self.IID_COL].isin(item_set)].reset_index(drop=True)
        pnt(f'processed items down to {len(items)} public-split items')
        return items

    def _save_meta(self, formatted_meta):
        paths = self._paths()
        meta = {
            'version': self.VER,
            'stage': 'processed',
            'dataset': self.get_name(),
            'formatted_dir': str(self.formatted_dir),
            'num_test': int(self.num_test),
            'num_finetune': int(self.num_finetune),
            'valid_ratio': float(self.VALID_RATIO),
            'user_order_seed': self.user_order_seed,
            'item_col': self.IID_COL,
            'user_col': self.UID_COL,
            'history_col': self.HIS_COL,
            'default_attrs': list(self.default_attrs),
            'require_stringify': bool(self.REQUIRE_STRINGIFY),
            'formatted_meta_path': str(self._formatted_paths()['meta']),
            'formatted_version': formatted_meta.get('version'),
            'provides_test_set': bool(self.provides_test_set),
            'multi_item_col': self.multi_item_col,
            'use_all_users_in_processor': bool(self.use_all_users_in_processor),
            'split_ratio': float(self.split_ratio),
            'remaining_users_as_valid': bool(self.remaining_users_as_valid),
        }
        paths['meta'].write_text(json.dumps(meta, indent=2) + '\n')

    def _save_stats(self):
        paths = self._paths()
        stats = {
            'processed_item_count': int(len(self.items)),
            'formatted_user_count': int(len(self.users)),
            'valid_user_count': int(len(self.valid_set)) if self.valid_set is not None else 0,
            'test_user_count': int(len(self.test_set)) if self.test_set is not None else 0,
            'finetune_user_count': int(len(self.finetune_set)) if self.finetune_set is not None else 0,
        }
        paths['stats'].write_text(json.dumps(stats, indent=2) + '\n')

    def _split_valid_from_test(self, test_set: pd.DataFrame):
        valid_count = int(len(test_set) * self.VALID_RATIO)
        if len(test_set) > 0 and valid_count <= 0:
            valid_count = 1
        valid_set = test_set.iloc[:valid_count].reset_index(drop=True)
        remain_test = test_set.iloc[valid_count:].reset_index(drop=True)
        return valid_set, remain_test

    def _resolve_finetune_count_from_ratio(self, total_count: int):
        if not 0.0 < float(self.split_ratio) < 1.0:
            raise ValueError(f'split_ratio must be in (0, 1), got {self.split_ratio}')
        finetune_count = int(total_count * self.split_ratio)
        if total_count > 1:
            finetune_count = min(max(finetune_count, 1), total_count - 1)
        return finetune_count

    def load_public_sets(self):
        paths = self._paths()
        if self.processed_valid:
            pnt(f'loading processed {self.get_name()} splits from cache')
            self._load_processed_meta()
            self.items = pd.read_parquet(paths['items'])
            if self.valid_set_required:
                self.valid_set = pd.read_parquet(paths['valid'])
            if self.test_set_required:
                self.test_set = pd.read_parquet(paths['test'])
            if self.finetune_set_required:
                self.finetune_set = pd.read_parquet(paths['finetune'])
            return

        formatted_meta = self.load_formatted()
        pnt(f'processing {self.get_name()} public splits from formatted users')
        users_order = self._load_user_order()
        iterator = self._iterator(users_order, self.users)

        if self.use_all_users_in_processor:
            ordered_users = self._split(iterator, len(users_order)).reset_index(drop=True)
            finetune_count = self._resolve_finetune_count_from_ratio(len(ordered_users))
            raw_test_set = ordered_users.iloc[finetune_count:].reset_index(drop=True)
            self.finetune_set = ordered_users.iloc[:finetune_count].reset_index(drop=True)
            if self.provides_test_set and self.remaining_users_as_valid:
                self.valid_set = raw_test_set
                self.test_set = self.formatted_test_set.reset_index(drop=True)
            else:
                self.valid_set, self.test_set = self._split_valid_from_test(raw_test_set)
            if self.provides_test_set:
                self.test_set = self.formatted_test_set.reset_index(drop=True)
            self.valid_set.to_parquet(paths['valid'], index=False)
            self.test_set.to_parquet(paths['test'], index=False)
            self.finetune_set.to_parquet(paths['finetune'], index=False)
            if self.provides_test_set:
                pnt(
                    f'generated full-user train/valid splits for {self.get_name()} '
                    f'total={len(ordered_users)} split_ratio={self.split_ratio:g} '
                    f'finetune={len(self.finetune_set)} valid={len(self.valid_set)} '
                    f'test=official({len(self.test_set)})'
                )
            else:
                pnt(
                    f'generated full-user splits for {self.get_name()} '
                    f'total={len(ordered_users)} split_ratio={self.split_ratio:g} '
                    f'finetune={len(self.finetune_set)} valid={len(self.valid_set)} test={len(self.test_set)}'
                )
        elif self.provides_test_set:
            if self.valid_set_required:
                self.valid_set = self._split(iterator, self.num_test)
                self.valid_set.reset_index(drop=True, inplace=True)
                self.valid_set.to_parquet(paths['valid'], index=False)
                pnt(f'generated valid set with {len(self.valid_set)}/{self.num_test} samples from train users')
            if self.test_set_required:
                self.test_set = self.formatted_test_set.reset_index(drop=True)
                self.test_set.to_parquet(paths['test'], index=False)
                pnt(f'using formatter-provided official test set with {len(self.test_set)} samples')
        elif self.test_set_required:
            raw_test_set = self._split(iterator, self.num_test)
            raw_test_set.reset_index(drop=True, inplace=True)
            self.valid_set, self.test_set = self._split_valid_from_test(raw_test_set)
            self.valid_set.to_parquet(paths['valid'], index=False)
            self.test_set.to_parquet(paths['test'], index=False)
            pnt(
                f'generated valid/test sets from {len(raw_test_set)}/{self.num_test} public users '
                f'valid={len(self.valid_set)} test={len(self.test_set)} ratio={self.VALID_RATIO:g}'
            )

        if self.finetune_set_required and not self.use_all_users_in_processor:
            self.finetune_set = self._split(iterator, self.num_finetune)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.finetune_set.to_parquet(paths['finetune'], index=False)
            pnt(f'generated finetune set with {len(self.finetune_set)}/{self.num_finetune} samples')

        self.items = self._build_processed_items()
        self.items.to_parquet(paths['items'], index=False)
        self._save_meta(formatted_meta)
        self._save_stats()

    def load(self):
        self.load_public_sets()

        if self.IID_COL is None:
            self._load_processed_meta()

        self.items = self._stringify(self.items)
        if self.test_set is not None:
            self.test_set = self._stringify(self.test_set)
        if self.valid_set is not None:
            self.valid_set = self._stringify(self.valid_set)
        if self.finetune_set is not None:
            self.finetune_set = self._stringify(self.finetune_set)

        if self.REQUIRE_STRINGIFY:
            if self.valid_set is not None:
                self.valid_set[self.HIS_COL] = self.valid_set[self.HIS_COL].apply(lambda x: [str(item) for item in x])
            if self.test_set is not None:
                self.test_set[self.HIS_COL] = self.test_set[self.HIS_COL].apply(lambda x: [str(item) for item in x])
            if self.finetune_set is not None:
                self.finetune_set[self.HIS_COL] = self.finetune_set[self.HIS_COL].apply(lambda x: [str(item) for item in x])

        self.item_vocab = dict(zip(self.items[self.IID_COL], range(len(self.items))))
        self.user_vocab = None if self.users is None else dict(zip(self.users[self.UID_COL], range(len(self.users))))
        self._loaded = True
        return self
