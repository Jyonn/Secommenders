import abc
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pigmento import pnt
from tqdm import tqdm

from formatters.base_formatter import BaseFormatter
from utils.stable_random import stable_shuffle


class RecIFVideoFormatter(BaseFormatter):
    VER = 'v1.1'

    IID_COL = 'pid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    DOMAIN = 'video'
    RAW_HISTORY_COL = 'hist_video_pid'
    OFFICIAL_TEST_DIR = 'video'
    OFFICIAL_TEST_HISTORY_COL = 'hist_pid'

    REQUIRE_STRINGIFY = False

    FILTER_ROUNDS = 3
    DEFAULT_N_CORE = 20
    DEFAULT_MIN_LENGTH = 10
    DEFAULT_MAX_LENGTH = 50

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self.n_core = int(self.DEFAULT_N_CORE)
        self.min_length = int(self.DEFAULT_MIN_LENGTH)
        self.max_length = int(self.DEFAULT_MAX_LENGTH)

        if self.n_core <= 0:
            raise ValueError('DEFAULT_N_CORE must be positive')
        if self.min_length <= 0:
            raise ValueError('DEFAULT_MIN_LENGTH must be positive')
        if self.max_length < self.min_length:
            raise ValueError('DEFAULT_MAX_LENGTH must be >= DEFAULT_MIN_LENGTH')

        self._filtered_items: pd.DataFrame | None = None
        self._filtered_users: pd.DataFrame | None = None

    @property
    def default_attrs(self):
        return ['caption']

    @property
    def source_path(self) -> Path:
        return Path(self.data_dir) / 'onerec_bench_release.parquet'

    @property
    def caption_path(self) -> Path:
        return Path(self.data_dir) / 'pid2caption.parquet'

    @property
    def official_video_test_path(self) -> Path:
        return Path(self.data_dir) / 'benchmark_data' / self.OFFICIAL_TEST_DIR / f'{self.OFFICIAL_TEST_DIR}_test.parquet'

    def _save_meta(self):
        meta_path = self._paths()['meta']
        meta = {
            'version': self.VER,
            'stage': 'formatted',
            'dataset': self.get_name(),
            'user_order_seed': self.get_seed(),
            'data_dir': self.data_dir,
            'item_col': self.IID_COL,
            'user_col': self.UID_COL,
            'history_col': self.HIS_COL,
            'default_attrs': list(self.default_attrs),
            'require_stringify': bool(self.REQUIRE_STRINGIFY),
            'provides_test_set': bool(self.PROVIDES_TEST_SET),
            'multi_item_col': self.MULTI_ITEM_COL,
            'use_all_users_in_processor': bool(self.USE_ALL_USERS_IN_PROCESSOR),
            'split_ratio': float(self.SPLIT_RATIO),
        }
        meta.update(
            {
                'domain': self.DOMAIN,
                'raw_history_col': self.RAW_HISTORY_COL,
                'official_test_dir': self.OFFICIAL_TEST_DIR,
                'official_test_history_col': self.OFFICIAL_TEST_HISTORY_COL,
                'n_core': int(self.n_core),
                'min_length': int(self.min_length),
                'max_length': int(self.max_length),
                'filter_rounds': int(self.FILTER_ROUNDS),
                'filter_pipeline': ['caption-exists-once', 'n-core-and-length-three-rounds'],
            }
        )
        meta.update(self._extra_meta())
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    def _cache_meta_matches(self, cached_meta):
        return (
            cached_meta.get('domain') == self.DOMAIN
            and cached_meta.get('raw_history_col') == self.RAW_HISTORY_COL
            and int(cached_meta.get('n_core', -1)) == int(self.n_core)
            and int(cached_meta.get('min_length', -1)) == int(self.min_length)
            and int(cached_meta.get('max_length', -1)) == int(self.max_length)
            and int(cached_meta.get('filter_rounds', -1)) == int(self.FILTER_ROUNDS)
        )

    @staticmethod
    def _normalize_history(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    def _normalize_item_id(self, value):
        return value

    def _normalize_history_ids(self, history):
        return [self._normalize_item_id(item) for item in self._normalize_history(history)]

    def _load_raw_users(self) -> pd.DataFrame:
        pnt(f'loading RecIF {self.DOMAIN} histories from {self.source_path}')
        users = pd.read_parquet(self.source_path, columns=[self.UID_COL, self.RAW_HISTORY_COL])
        users = users.rename(columns={self.RAW_HISTORY_COL: self.HIS_COL})
        users[self.HIS_COL] = users[self.HIS_COL].apply(self._normalize_history_ids)
        users = users[users[self.HIS_COL].map(len) > 0].reset_index(drop=True)
        pnt(f'loaded {len(users)} raw RecIF {self.DOMAIN} users with non-empty history')
        return users

    def _stream_caption_pid_set(self) -> set:
        pnt(f'scanning caption pid coverage from {self.caption_path}')
        parquet_file = pq.ParquetFile(self.caption_path)
        caption_pid_set = set()
        progress = tqdm(
            parquet_file.iter_batches(columns=[self.IID_COL], batch_size=200_000),
            desc='caption-pids',
        )
        for batch in progress:
            pid_array = batch.column(0).to_numpy()
            caption_pid_set.update(self._normalize_item_id(pid) for pid in pid_array.tolist())
        pnt(f'caption coverage collected for {len(caption_pid_set)} pids')
        return caption_pid_set

    def _apply_allowed_item_filter(self, users: pd.DataFrame, allowed_items: set, desc: str) -> pd.DataFrame:
        filtered_histories = []
        kept_uids = []
        iterator = zip(users[self.UID_COL].tolist(), users[self.HIS_COL].tolist())
        for uid, history in tqdm(iterator, total=len(users), desc=desc):
            filtered = [pid for pid in history if pid in allowed_items]
            if filtered:
                kept_uids.append(uid)
                filtered_histories.append(filtered)
        return pd.DataFrame({self.UID_COL: kept_uids, self.HIS_COL: filtered_histories})

    def _tail_history(self, history: list):
        if len(history) > self.max_length:
            return history[-self.max_length:]
        return history

    def _apply_length_constraints(self, users: pd.DataFrame, desc: str) -> pd.DataFrame:
        trimmed_histories = []
        kept_uids = []
        iterator = zip(users[self.UID_COL].tolist(), users[self.HIS_COL].tolist())
        for uid, history in tqdm(iterator, total=len(users), desc=desc):
            history = self._tail_history(history)
            if len(history) < self.min_length:
                continue
            kept_uids.append(uid)
            trimmed_histories.append(history)
        return pd.DataFrame({self.UID_COL: kept_uids, self.HIS_COL: trimmed_histories})

    @staticmethod
    def _count_items(histories: Iterable[list]) -> Counter:
        counter = Counter()
        for history in histories:
            counter.update(history)
        return counter

    def _load_caption_rows(self, final_item_ids: set) -> pd.DataFrame:
        pnt(f'loading captions for {len(final_item_ids)} filtered {self.DOMAIN} items from {self.caption_path}')
        parquet_file = pq.ParquetFile(self.caption_path)
        rows = []
        for batch in tqdm(
            parquet_file.iter_batches(columns=[self.IID_COL, 'dense_caption'], batch_size=100_000),
            desc='captions',
        ):
            frame = batch.to_pandas()
            frame = frame[frame[self.IID_COL].isin(final_item_ids)]
            if not frame.empty:
                rows.append(frame)
        if not rows:
            raise ValueError(f'No RecIF {self.DOMAIN} items with captions survived filtering')
        items = pd.concat(rows, ignore_index=True)
        items = items.rename(columns={'dense_caption': 'caption'})
        items[self.IID_COL] = items[self.IID_COL].apply(self._normalize_item_id)
        items = items.drop_duplicates(subset=[self.IID_COL], keep='first').reset_index(drop=True)
        return items[[self.IID_COL, 'caption']]

    def _run_filter_pipeline(self):
        if self._filtered_items is not None and self._filtered_users is not None:
            return

        pnt(
            'RecIF video filter settings: '
            f'n_core={self.n_core}, min_length={self.min_length}, max_length={self.max_length}, '
            f'rounds={self.FILTER_ROUNDS}'
        )

        users = self._load_raw_users()

        caption_pid_set = self._stream_caption_pid_set()
        users = self._apply_allowed_item_filter(users, caption_pid_set, desc='caption-filter')
        users = self._apply_length_constraints(users, desc='caption-length')

        for round_index in range(self.FILTER_ROUNDS):
            counts = self._count_items(users[self.HIS_COL].tolist())
            allowed = {pid for pid, count in counts.items() if count >= self.n_core}
            pnt(
                f'round {round_index + 1}/{self.FILTER_ROUNDS}: '
                f'{len(allowed)} items survive n_core>={self.n_core}'
            )
            users = self._apply_allowed_item_filter(users, allowed, desc=f'ncore-filter@{round_index + 1}')
            users = self._apply_length_constraints(users, desc=f'length-filter@{round_index + 1}')

        final_item_ids = set()
        for history in users[self.HIS_COL].tolist():
            final_item_ids.update(history)

        items = self._load_caption_rows(final_item_ids)
        final_item_set = set(items[self.IID_COL].tolist())
        users = self._apply_allowed_item_filter(users, final_item_set, desc='final-caption-align')
        users = self._apply_length_constraints(users, desc='final-length-align')

        self._filtered_items = items.sort_values(self.IID_COL).reset_index(drop=True)
        self._filtered_users = users.reset_index(drop=True)

        pnt(
            f'RecIF {self.DOMAIN} formatting complete with items={len(self._filtered_items)} '
            f'users={len(self._filtered_users)}'
        )

    def load_items(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return cast(pd.DataFrame, self._filtered_items)

    def load_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return self._filtered_users

    def _load_official_video_test_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        final_item_set = set(cast(pd.DataFrame, self._filtered_items)[self.IID_COL].tolist())
        test_frame = pd.read_parquet(self.official_video_test_path, columns=[self.OFFICIAL_TEST_HISTORY_COL, 'metadata'])

        records = []
        for index, row in tqdm(test_frame.iterrows(), total=len(test_frame), desc=f'official-{self.DOMAIN}-test'):
            history = self._normalize_history_ids(row[self.OFFICIAL_TEST_HISTORY_COL])
            history = [pid for pid in history if pid in final_item_set]
            history = self._tail_history(history)
            if not history:
                continue

            metadata = row['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            elif not isinstance(metadata, dict):
                metadata = {}
            answer_pids = self._normalize_history_ids(metadata.get('answer_pid'))
            answer_pids = [pid for pid in answer_pids if pid in final_item_set]
            if not answer_pids:
                continue

            records.append(
                {
                    self.UID_COL: f'official-{self.DOMAIN}-test-{index}',
                    self.HIS_COL: history,
                    'answer_pids': answer_pids,
                }
            )
        if not records:
            raise ValueError(f'No official RecIF {self.DOMAIN} test samples survived item filtering')
        return pd.DataFrame(records)


class RecIFVideoLargeFormatter(RecIFVideoFormatter):
    DEFAULT_N_CORE = 10
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50


class RecIFVideoXLargeFormatter(RecIFVideoFormatter):
    DEFAULT_N_CORE = 3
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50


class RecIFVideoAllFormatter(RecIFVideoFormatter):
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9


class RecIFVideoLargeAllFormatter(RecIFVideoLargeFormatter):
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9

class RecIFVideoXLargeAllFormatter(RecIFVideoXLargeFormatter):
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9


class RecIFVideoXLargeAllOfficialFormatter(RecIFVideoXLargeFormatter):
    PROVIDES_TEST_SET = True
    MULTI_ITEM_COL = 'answer_pids'
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9

    @classmethod
    def get_seed(cls):
        return RecIFVideoXLargeAllFormatter.get_seed()

    def load_test_users(self) -> pd.DataFrame:
        return self._load_official_video_test_users()


class RecIFAdsFormatter(RecIFVideoFormatter, abc.ABC):
    DOMAIN = 'ad'
    RAW_HISTORY_COL = 'hist_ad_pid'
    OFFICIAL_TEST_DIR = 'ad'
    OFFICIAL_TEST_HISTORY_COL = 'hist_ad'
    PROVIDES_TEST_SET = True
    MULTI_ITEM_COL = 'answer_pids'
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9

    @classmethod
    @abc.abstractmethod
    def get_seed(cls):
        raise NotImplementedError

    def _normalize_item_id(self, value):
        if pd.isna(value):
            return value
        return int(value)

    def load_test_users(self) -> pd.DataFrame:
        return self._load_official_video_test_users()


class RecIFAdsAllFormatter(RecIFAdsFormatter):
    @classmethod
    def get_seed(cls):
        return 'recifadsall'


class RecIFAdsLargeAllFormatter(RecIFAdsFormatter):
    DEFAULT_N_CORE = 10
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50

    @classmethod
    def get_seed(cls):
        return 'recifadslargeall'


class RecIFAdsXLargeAllFormatter(RecIFAdsFormatter):
    DEFAULT_N_CORE = 3
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50

    @classmethod
    def get_seed(cls):
        return 'recifadsxlargeall'


class RecIFScaleFormatter(RecIFVideoFormatter, abc.ABC):
    VER = 'v1.2-scale'

    PROVIDES_TEST_SET = True
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9

    SCALE_TEST_RATIO = 0.003
    SCALE_SHUFFLE_SEED = 'RECIF'
    SCALE_SHUFFLE_VERSION = 'v1'

    DEFAULT_N_CORE = 20
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self._filtered_test_users: pd.DataFrame | None = None
        self._scale_total_sequences: int | None = None
        self._scale_test_start: int | None = None
        self._scale_train_limit: int | None = None
        self._scale_reused_from: str | None = None

    @classmethod
    @abc.abstractmethod
    def scale_percent(cls) -> int:
        raise NotImplementedError

    @classmethod
    def get_seed(cls):
        return cls.SCALE_SHUFFLE_SEED

    def _extra_meta(self):
        meta = {
            'scale_percent': int(self.scale_percent()),
            'scale_test_ratio': float(self.SCALE_TEST_RATIO),
            'scale_shuffle_seed': self.get_seed(),
            'scale_shuffle_version': self.SCALE_SHUFFLE_VERSION,
            'scale_split_policy': 'stable-shuffle-prefix-train-tail-test',
            'filter_pipeline': [
                'caption-exists',
                'repeat-three-rounds:item-n-core-and-non-overlap-max-length-chunks',
                'stable-shuffle-sequences',
                'prefix-scale-train-tail-test',
            ],
        }
        if self._scale_total_sequences is not None:
            meta.update(
                {
                    'scale_total_sequences': int(self._scale_total_sequences),
                    'scale_test_start': int(self._scale_test_start),
                    'scale_train_limit': int(self._scale_train_limit),
                }
            )
        return meta

    def _cache_meta_matches(self, cached_meta):
        return (
            super()._cache_meta_matches(cached_meta)
            and int(cached_meta.get('scale_percent', -1)) == int(self.scale_percent())
            and float(cached_meta.get('scale_test_ratio', -1.0)) == float(self.SCALE_TEST_RATIO)
            and cached_meta.get('scale_shuffle_seed') == self.get_seed()
            and cached_meta.get('scale_shuffle_version') == self.SCALE_SHUFFLE_VERSION
            and cached_meta.get('scale_total_sequences') is not None
            and cached_meta.get('scale_test_start') is not None
            and cached_meta.get('scale_train_limit') is not None
        )

    @classmethod
    def _scale_dataset_prefix(cls):
        return 'ra' if cls.DOMAIN == 'ad' else 'rv'

    @classmethod
    def _parse_scale_dataset(cls, dataset: str):
        prefix = cls._scale_dataset_prefix()
        dataset = str(dataset).lower()
        if not dataset.startswith(prefix):
            return None
        suffix = dataset[len(prefix):]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _scale_split_points(self, total_sequences: int):
        test_start = int(total_sequences * (1.0 - self.SCALE_TEST_RATIO))
        if total_sequences > 1:
            test_start = min(max(test_start, 1), total_sequences - 1)
        train_limit = int(total_sequences * (self.scale_percent() / 100.0))
        train_limit = min(train_limit, test_start)
        return test_start, train_limit

    def _scale_meta_compatible(self, meta: dict):
        return (
            meta.get('version') == self.VER
            and meta.get('domain') == self.DOMAIN
            and meta.get('raw_history_col') == self.RAW_HISTORY_COL
            and str(meta.get('data_dir')) == str(self.data_dir)
            and int(meta.get('n_core', -1)) == int(self.n_core)
            and int(meta.get('min_length', -1)) == int(self.min_length)
            and int(meta.get('max_length', -1)) == int(self.max_length)
            and int(meta.get('filter_rounds', -1)) == int(self.FILTER_ROUNDS)
            and float(meta.get('scale_test_ratio', -1.0)) == float(self.SCALE_TEST_RATIO)
            and meta.get('scale_shuffle_seed') == self.get_seed()
            and meta.get('scale_shuffle_version') == self.SCALE_SHUFFLE_VERSION
            and meta.get('scale_total_sequences') is not None
            and meta.get('scale_test_start') is not None
        )

    @staticmethod
    def _safe_int(value):
        if value is None or pd.isna(value):
            return None
        return int(value)

    @staticmethod
    def _safe_str(value):
        if value is None or pd.isna(value):
            return None
        return str(value)

    def _iter_larger_scale_candidates(self):
        target_percent = int(self.scale_percent())
        formatted_root = Path(self.store_dir).parent
        prefix = self._scale_dataset_prefix()
        candidates = []
        for meta_path in formatted_root.glob(f'{prefix}*/meta.json'):
            dataset = meta_path.parent.name.lower()
            if dataset == self.get_name():
                continue
            scale_percent = self._parse_scale_dataset(dataset)
            if scale_percent is None or scale_percent <= target_percent:
                continue
            candidates.append((scale_percent, dataset, meta_path))
        return sorted(candidates, key=lambda item: item[0])

    def _records_to_users_from_frame(self, frame: pd.DataFrame, split: str):
        rows = []
        for index, row in enumerate(frame.to_dict('records')):
            history = self._normalize_history_ids(row[self.HIS_COL])
            source_uid = self._safe_str(row.get('source_uid'))
            if source_uid is None:
                source_uid = self._safe_str(row.get(self.UID_COL))
            output_row = {
                self.UID_COL: f'{self.get_name()}-{split}-{index:08d}',
                self.HIS_COL: history,
                'source_uid': source_uid,
                'segment_index': index,
                'segment_length': len(history),
            }
            if 'global_sequence_index' in row:
                global_sequence_index = self._safe_int(row['global_sequence_index'])
                if global_sequence_index is not None:
                    output_row['global_sequence_index'] = global_sequence_index
            rows.append(output_row)
        return pd.DataFrame(rows)

    def _try_load_from_larger_scale(self):
        for source_percent, dataset, meta_path in self._iter_larger_scale_candidates():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            if not self._scale_meta_compatible(meta):
                continue
            if int(meta.get('scale_percent', -1)) != int(source_percent):
                continue

            source_dir = meta_path.parent
            items_path = source_dir / 'items.parquet'
            users_path = source_dir / 'users.parquet'
            test_users_path = source_dir / 'test_users.parquet'
            if not (items_path.exists() and users_path.exists() and test_users_path.exists()):
                continue

            total_sequences = int(meta['scale_total_sequences'])
            test_start, train_limit = self._scale_split_points(total_sequences)
            if train_limit <= 0:
                continue

            users = pd.read_parquet(users_path)
            test_users = pd.read_parquet(test_users_path)
            if len(users) < train_limit:
                continue

            source_train_limit = int(meta.get('scale_train_limit', len(users)))
            source_test_start = int(meta.get('scale_test_start'))
            if (
                source_train_limit < train_limit
                or source_test_start != test_start
                or len(test_users) != total_sequences - test_start
            ):
                continue

            self._filtered_items = pd.read_parquet(items_path).reset_index(drop=True)
            self._filtered_users = self._records_to_users_from_frame(users.iloc[:train_limit], split='train')
            self._filtered_test_users = self._records_to_users_from_frame(test_users, split='test')
            self._scale_total_sequences = total_sequences
            self._scale_test_start = test_start
            self._scale_train_limit = train_limit
            self._scale_reused_from = dataset
            pnt(
                f'RecIF {self.DOMAIN} scale reused formatted {dataset} ({source_percent}%) '
                f'for {self.get_name()} ({self.scale_percent()}%) train_sequences={train_limit} '
                f'test_sequences={len(self._filtered_test_users)} total_sequences={total_sequences}'
            )
            return True
        return False

    def _split_history_chunks(self, history: list):
        for start in range(0, len(history), self.max_length):
            chunk = history[start:start + self.max_length]
            if len(chunk) >= self.min_length:
                yield chunk

    def _users_to_sequence_records(self, users: pd.DataFrame):
        records = []
        iterator = zip(users[self.UID_COL].tolist(), users[self.HIS_COL].tolist())
        for uid, history in tqdm(iterator, total=len(users), desc='seed-sequences'):
            normalized = self._normalize_history_ids(history)
            for chunk in self._split_history_chunks(normalized):
                records.append({'source_uid': uid, self.HIS_COL: chunk})
        return records

    def _filter_and_chunk_records(self, records: list[dict], allowed_items: set, desc: str):
        filtered_records = []
        for record in tqdm(records, total=len(records), desc=desc):
            history = [pid for pid in record[self.HIS_COL] if pid in allowed_items]
            for chunk in self._split_history_chunks(history):
                filtered_records.append({'source_uid': record['source_uid'], self.HIS_COL: chunk})
        return filtered_records

    def _records_item_set(self, records: list[dict]) -> set:
        item_set = set()
        for record in records:
            item_set.update(record[self.HIS_COL])
        return item_set

    def _records_to_users(self, records: list[dict], split: str):
        rows = []
        for index, record in enumerate(records):
            row = {
                self.UID_COL: f'{self.get_name()}-{split}-{index:08d}',
                self.HIS_COL: record[self.HIS_COL],
                'source_uid': str(record['source_uid']),
                'segment_index': index,
                'segment_length': len(record[self.HIS_COL]),
            }
            if 'global_sequence_index' in record:
                global_sequence_index = self._safe_int(record['global_sequence_index'])
                if global_sequence_index is not None:
                    row['global_sequence_index'] = global_sequence_index
            rows.append(row)
        return pd.DataFrame(rows)

    def _run_filter_pipeline(self):
        if (
            self._filtered_items is not None
            and self._filtered_users is not None
            and self._filtered_test_users is not None
        ):
            return

        pnt(
            f'RecIF {self.DOMAIN} scale formatting settings: '
            f'scale={self.scale_percent()}%, n_core={self.n_core}, '
            f'min_length={self.min_length}, max_length={self.max_length}, rounds={self.FILTER_ROUNDS}, '
            f'test_ratio={self.SCALE_TEST_RATIO:g}'
        )

        if self._try_load_from_larger_scale():
            return

        users = self._load_raw_users()
        records = self._users_to_sequence_records(users)

        caption_pid_set = self._stream_caption_pid_set()
        records = self._filter_and_chunk_records(records, caption_pid_set, desc='caption-filter-sequences')
        if not records:
            raise ValueError(f'No RecIF {self.DOMAIN} scale sequences survived caption filtering')

        for round_index in range(self.FILTER_ROUNDS):
            counts = self._count_items(record[self.HIS_COL] for record in records)
            allowed = {pid for pid, count in counts.items() if count >= self.n_core}
            pnt(
                f'scale round {round_index + 1}/{self.FILTER_ROUNDS}: '
                f'{len(allowed)} items survive n_core>={self.n_core}'
            )
            records = self._filter_and_chunk_records(
                records,
                allowed,
                desc=f'ncore-sequence-filter@{round_index + 1}',
            )
            if not records:
                raise ValueError(
                    f'No RecIF {self.DOMAIN} scale sequences survived round {round_index + 1}; '
                    'try a smaller n_core or min_length'
                )

        final_item_ids = self._records_item_set(records)
        items = self._load_caption_rows(final_item_ids)
        final_item_set = set(items[self.IID_COL].tolist())
        records = self._filter_and_chunk_records(records, final_item_set, desc='final-caption-align-sequences')
        if not records:
            raise ValueError(f'No RecIF {self.DOMAIN} scale sequences survived final item alignment')

        records = stable_shuffle(records, seed=self.get_seed())
        for index, record in enumerate(records):
            record['global_sequence_index'] = index
        test_start, train_limit = self._scale_split_points(len(records))
        if train_limit <= 0:
            raise ValueError(f'scale={self.scale_percent()}% produced no train sequences from {len(records)} records')

        train_records = records[:train_limit]
        test_records = records[test_start:]
        if not test_records:
            raise ValueError(f'scale split produced no test sequences from {len(records)} records')

        self._filtered_items = items.sort_values(self.IID_COL).reset_index(drop=True)
        self._filtered_users = self._records_to_users(train_records, split='train')
        self._filtered_test_users = self._records_to_users(test_records, split='test')
        self._scale_total_sequences = len(records)
        self._scale_test_start = test_start
        self._scale_train_limit = train_limit

        pnt(
            f'RecIF {self.DOMAIN} scale formatting complete with items={len(self._filtered_items)} '
            f'train_sequences={len(self._filtered_users)} test_sequences={len(self._filtered_test_users)} '
            f'total_sequences={len(records)}'
        )

    def load_test_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return cast(pd.DataFrame, self._filtered_test_users)


class RecIFVideoScaleFormatter(RecIFScaleFormatter, abc.ABC):
    DOMAIN = 'video'
    RAW_HISTORY_COL = 'hist_video_pid'
    OFFICIAL_TEST_DIR = 'video'
    OFFICIAL_TEST_HISTORY_COL = 'hist_pid'


class RecIFAdsScaleFormatter(RecIFScaleFormatter, abc.ABC):
    DOMAIN = 'ad'
    RAW_HISTORY_COL = 'hist_ad_pid'
    OFFICIAL_TEST_DIR = 'ad'
    OFFICIAL_TEST_HISTORY_COL = 'hist_ad'

    def _normalize_item_id(self, value):
        if pd.isna(value):
            return value
        return int(value)


class RV1Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 1


class RV2Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 2


class RV5Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 5


class RV10Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 10


class RV20Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 20


class RV30Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 30


class RV40Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 40


class RV50Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 50


class RV60Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 60


class RV70Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 70


class RV80Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 80


class RV90Formatter(RecIFVideoScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 90


class RA1Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 1


class RA2Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 2


class RA5Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 5


class RA10Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 10


class RA20Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 20


class RA30Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 30


class RA40Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 40


class RA50Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 50


class RA60Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 60


class RA70Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 70


class RA80Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 80


class RA90Formatter(RecIFAdsScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 90
