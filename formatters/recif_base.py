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


class RecIFBaseFormatter(BaseFormatter):
    VER = 'v1.1'

    IID_COL = 'pid'
    UID_COL = 'uid'
    HIS_COL = 'history'

    DOMAIN: str
    RAW_HISTORY_COL: str
    OFFICIAL_TEST_DIR: str
    OFFICIAL_TEST_HISTORY_COL: str

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
    def official_test_path(self) -> Path:
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
            f'RecIF {self.DOMAIN} filter settings: '
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
        return cast(pd.DataFrame, self._filtered_users)

    def _load_official_test_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        final_item_set = set(cast(pd.DataFrame, self._filtered_items)[self.IID_COL].tolist())
        test_frame = pd.read_parquet(self.official_test_path, columns=[self.OFFICIAL_TEST_HISTORY_COL, 'metadata'])

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


class RecIFFullFormatterMixin:
    VER = 'v1.3-full'
    PROVIDES_TEST_SET = True
    MULTI_ITEM_COL = None
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 99.0 / 99.5

    FULL_TEST_RATIO = 0.005
    FULL_SHUFFLE_SEED = 'RECIF'
    FULL_SHUFFLE_VERSION = 'v1'

    FILTER_ROUNDS = 0
    DEFAULT_N_CORE = 1
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 256

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self._full_stats: dict = {}
        self._filtered_test_users: pd.DataFrame | None = None

    def _extra_meta(self):
        return {
            'formatter_mode': 'full',
            'min_history_length': int(self.min_length),
            'max_history_length': int(self.max_length),
            'item_filter_policy': 'no-n-core',
            'history_policy': 'latest-items',
            'full_test_ratio': float(self.FULL_TEST_RATIO),
            'full_shuffle_seed': self.FULL_SHUFFLE_SEED,
            'full_shuffle_version': self.FULL_SHUFFLE_VERSION,
            'full_split_policy': 'stable-shuffle-train-tail-test',
            'remaining_users_as_valid': True,
            'filter_pipeline': [
                'caption-exists-once',
                'latest-256',
                'min-length-5',
                'stable-shuffle-sequences',
                'tail-test',
            ],
        }

    def _cache_meta_matches(self, cached_meta):
        return (
            super()._cache_meta_matches(cached_meta)
            and float(cached_meta.get('full_test_ratio', -1.0)) == float(self.FULL_TEST_RATIO)
            and cached_meta.get('full_shuffle_seed') == self.FULL_SHUFFLE_SEED
            and cached_meta.get('full_shuffle_version') == self.FULL_SHUFFLE_VERSION
        )

    def _extra_stats(self):
        return dict(self._full_stats)

    def _history_item_set(self, users: pd.DataFrame) -> set:
        item_ids = set()
        for history in users[self.HIS_COL].tolist():
            item_ids.update(item for item in history if not pd.isna(item))
        return item_ids

    def _run_filter_pipeline(self):
        if self._filtered_items is not None and self._filtered_users is not None:
            return

        pnt(
            f'RecIF {self.DOMAIN} full formatting settings: '
            f'min_length={self.min_length}, max_length={self.max_length}, item_n_core=disabled'
        )

        raw_users = self._load_raw_users()
        raw_histories = raw_users[self.HIS_COL].tolist()
        item_count_before = int(len(self._history_item_set(raw_users)))
        user_count_before = int(raw_users[self.UID_COL].dropna().nunique())
        interaction_count_before = int(sum(len(history) for history in raw_histories))
        sequence_lengths_before = self._history_length_stats(raw_histories)

        caption_pid_set = self._stream_caption_pid_set()
        users = self._apply_allowed_item_filter(raw_users, caption_pid_set, desc='caption-filter')
        users = self._apply_length_constraints(users, desc='latest-length')

        final_item_ids = self._history_item_set(users)
        items = self._load_caption_rows(final_item_ids)
        final_item_set = set(items[self.IID_COL].tolist())
        users = self._apply_allowed_item_filter(users, final_item_set, desc='final-caption-align')
        users = self._apply_length_constraints(users, desc='final-length-align')

        final_item_ids = self._history_item_set(users)
        items = items[items[self.IID_COL].isin(final_item_ids)].reset_index(drop=True)

        users = pd.DataFrame(stable_shuffle(users.to_dict('records'), seed=self.FULL_SHUFFLE_SEED))
        test_start = int(len(users) * (1.0 - self.FULL_TEST_RATIO))
        if len(users) > 1:
            test_start = min(max(test_start, 1), len(users) - 1)
        train_users = users.iloc[:test_start].reset_index(drop=True)
        test_users = users.iloc[test_start:].reset_index(drop=True)
        if train_users.empty or test_users.empty:
            raise ValueError(
                f'RecIF {self.DOMAIN} full split requires at least two filtered users; got {len(users)}'
            )

        interaction_count_after = int(users[self.HIS_COL].map(len).sum())
        sequence_lengths_after = self._history_length_stats(users[self.HIS_COL].tolist())
        self._full_stats = {
            'formatter_mode': 'full',
            'min_history_length_limit': int(self.min_length),
            'max_history_length_limit': int(self.max_length),
            'item_filter_policy': 'no-n-core; keep items observed in final user histories',
            'full_test_ratio': float(self.FULL_TEST_RATIO),
            'full_train_user_count': int(len(train_users)),
            'full_test_user_count': int(len(test_users)),
            'caption_item_count': int(len(caption_pid_set)),
            'item_count_before': item_count_before,
            'item_count_after': int(len(items)),
            'item_count_removed': int(item_count_before - len(items)),
            'user_count_before': user_count_before,
            'user_count_after': int(len(users)),
            'user_count_removed': int(user_count_before - len(users)),
            'raw_user_row_count': int(len(raw_users)),
            'interaction_count_before': interaction_count_before,
            'interaction_count_after': interaction_count_after,
            'interaction_count_removed': int(interaction_count_before - interaction_count_after),
            'user_sequence_length_before': sequence_lengths_before,
            'user_sequence_length_after': sequence_lengths_after,
        }

        self._filtered_items = items.sort_values(self.IID_COL).reset_index(drop=True)
        self._filtered_users = train_users
        self._filtered_test_users = test_users

        pnt(
            f'RecIF {self.DOMAIN} full count transition: items {item_count_before}->{len(items)} '
            f'users {user_count_before}->{len(users)} '
            f'interactions {interaction_count_before}->{interaction_count_after}'
        )
        self._print_history_length_stats(f'RecIF {self.DOMAIN} full before', sequence_lengths_before)
        self._print_history_length_stats(f'RecIF {self.DOMAIN} full after', sequence_lengths_after)
        pnt(
            f'RecIF {self.DOMAIN} full formatting complete with items={len(self._filtered_items)} '
            f'train_users={len(self._filtered_users)} test_users={len(self._filtered_test_users)} '
            f'interactions={interaction_count_after}'
        )

    def load_test_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return cast(pd.DataFrame, self._filtered_test_users)
