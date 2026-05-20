import json
import random
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from formatters.base_formatter import BaseFormatter
from utils.artifact import ArtifactStore


class Processor:
    VER = 'v2.0'

    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000

    def __init__(self, formatter: BaseFormatter, num_test=None, num_finetune=None):
        self.formatter = formatter

        self.IID_COL = formatter.IID_COL
        self.UID_COL = formatter.UID_COL
        self.HIS_COL = formatter.HIS_COL
        self.REQUIRE_STRINGIFY = formatter.REQUIRE_STRINGIFY

        self.num_test = self.NUM_TEST if num_test is None else num_test
        self.num_finetune = self.NUM_FINETUNE if num_finetune is None else num_finetune

        self.store = ArtifactStore(self.get_name())
        self.store_dir = str(self.store.processed_dir())
        self._loaded = False

        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

        self.test_set: Optional[pd.DataFrame] = None
        self.finetune_set: Optional[pd.DataFrame] = None

    def get_name(self):
        return self.formatter.get_name()

    @property
    def default_attrs(self):
        return self.formatter.default_attrs

    def _paths(self):
        base_dir = Path(self.store_dir)
        return {
            'items': base_dir / 'items.parquet',
            'test': base_dir / 'test.parquet',
            'finetune': base_dir / 'finetune.parquet',
            'user_order': base_dir / 'user_order.txt',
            'meta': base_dir / 'meta.json',
            'stats': base_dir / 'stats.json',
        }

    def _stringify(self, df: pd.DataFrame):
        return self.formatter._stringify(df)

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
        assert source in ['test', 'finetune', 'original'], 'source must be test, finetune, or original'
        return self.users if source == 'original' else getattr(self, f'{source}_set')

    def generate(self, slicer: Union[int, Callable], source='test', **kwargs):
        if not self._loaded:
            raise RuntimeError('Datasets not loaded')

        source_set = self.get_source_set(source)
        return self._iterate(source_set, slicer)

    def iterate(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='original')

    def test(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='test')

    def finetune(self, slicer: Union[int, Callable], **kwargs):
        return self.generate(slicer, source='finetune')

    def _iterator(self, user_order, users):
        users_by_id = users.set_index(self.UID_COL, drop=False)
        for uid in user_order:
            yield users_by_id.loc[uid]

    def _split(self, iterator, count):
        users = []
        for user in tqdm(iterator, total=count):
            users.append(user)
            if len(users) >= count:
                break
        return pd.DataFrame(users)

    def _load_user_order(self):
        path = self._paths()['user_order']
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]

        users = self.users[self.UID_COL].unique().tolist()
        random.shuffle(users)
        path.write_text(''.join(f'{user}\n' for user in users))
        return users

    @property
    def test_set_required(self):
        return self.num_test > 0

    @property
    def finetune_set_required(self):
        return self.num_finetune > 0

    @property
    def processed_valid(self):
        paths = self._paths()
        required = [paths['items']]
        if self.test_set_required:
            required.append(paths['test'])
        if self.finetune_set_required:
            required.append(paths['finetune'])
        return all(path.exists() for path in required)

    def _collect_public_item_set(self):
        if not self.test_set_required and not self.finetune_set_required:
            return set(self.formatter.items[self.IID_COL].unique())

        item_set = set()
        for dataframe in [self.test_set, self.finetune_set]:
            if dataframe is None:
                continue
            dataframe[self.HIS_COL].apply(lambda x: [item_set.add(i) for i in x])
        return item_set

    def _build_processed_items(self):
        item_set = self._collect_public_item_set()
        items = self.formatter.items[self.formatter.items[self.IID_COL].isin(item_set)].reset_index(drop=True)
        pnt(f'processed items down to {len(items)} public-split items')
        return items

    def _save_meta(self):
        paths = self._paths()
        meta = {
            'version': self.VER,
            'stage': 'processed',
            'dataset': self.get_name(),
            'formatter_dir': self.formatter.store_dir,
            'num_test': int(self.num_test),
            'num_finetune': int(self.num_finetune),
            'item_col': self.IID_COL,
            'user_col': self.UID_COL,
            'history_col': self.HIS_COL,
        }
        paths['meta'].write_text(json.dumps(meta, indent=2) + '\n')

    def _save_stats(self):
        paths = self._paths()
        stats = {
            'processed_item_count': int(len(self.items)),
            'formatted_user_count': int(len(self.users)),
            'test_user_count': int(len(self.test_set)) if self.test_set is not None else 0,
            'finetune_user_count': int(len(self.finetune_set)) if self.finetune_set is not None else 0,
        }
        paths['stats'].write_text(json.dumps(stats, indent=2) + '\n')

    def load_public_sets(self):
        paths = self._paths()
        if self.processed_valid:
            pnt(f'loading processed {self.get_name()} splits from cache')
            self.items = pd.read_parquet(paths['items'])
            if self.test_set_required:
                self.test_set = pd.read_parquet(paths['test'])
            if self.finetune_set_required:
                self.finetune_set = pd.read_parquet(paths['finetune'])
            return

        pnt(f'processing {self.get_name()} public splits from formatted users')
        users_order = self._load_user_order()
        iterator = self._iterator(users_order, self.users)

        if self.test_set_required:
            self.test_set = self._split(iterator, self.num_test)
            self.test_set.reset_index(drop=True, inplace=True)
            self.test_set.to_parquet(paths['test'], index=False)
            pnt(f'generated test set with {len(self.test_set)}/{self.num_test} samples')

        if self.finetune_set_required:
            self.finetune_set = self._split(iterator, self.num_finetune)
            self.finetune_set.reset_index(drop=True, inplace=True)
            self.finetune_set.to_parquet(paths['finetune'], index=False)
            pnt(f'generated finetune set with {len(self.finetune_set)}/{self.num_finetune} samples')

        self.items = self._build_processed_items()
        self.items.to_parquet(paths['items'], index=False)
        self._save_meta()
        self._save_stats()

    def load(self):
        self.formatter.load()
        self.users = self.formatter.users

        self.load_public_sets()

        self.items = self._stringify(self.items)
        if self.test_set is not None:
            self.test_set = self._stringify(self.test_set)
        if self.finetune_set is not None:
            self.finetune_set = self._stringify(self.finetune_set)

        if self.REQUIRE_STRINGIFY:
            self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(lambda x: [str(item) for item in x])
            if self.test_set is not None:
                self.test_set[self.HIS_COL] = self.test_set[self.HIS_COL].apply(lambda x: [str(item) for item in x])
            if self.finetune_set is not None:
                self.finetune_set[self.HIS_COL] = self.finetune_set[self.HIS_COL].apply(lambda x: [str(item) for item in x])

        self.item_vocab = dict(zip(self.items[self.IID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.UID_COL], range(len(self.users))))
        self._loaded = True
        return self
