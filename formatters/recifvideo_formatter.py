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


class RecIFVideoFormatter(BaseFormatter):
    VER = 'v1.0'

    IID_COL = 'pid'
    UID_COL = 'uid'
    HIS_COL = 'history'

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
        return Path(self.data_dir) / 'benchmark_data' / 'video' / 'video_test.parquet'

    def _save_meta(self):
        meta_path = self._paths()['meta']
        meta = {
            'version': self.VER,
            'stage': 'formatted',
            'dataset': self.get_name(),
            'data_dir': self.data_dir,
            'item_col': self.IID_COL,
            'user_col': self.UID_COL,
            'history_col': self.HIS_COL,
            'default_attrs': list(self.default_attrs),
            'require_stringify': bool(self.REQUIRE_STRINGIFY),
            'provides_test_set': bool(self.PROVIDES_TEST_SET),
        }
        meta.update(
            {
                'domain': 'video',
                'n_core': int(self.n_core),
                'min_length': int(self.min_length),
                'max_length': int(self.max_length),
                'filter_rounds': int(self.FILTER_ROUNDS),
                'filter_pipeline': ['caption-exists-once', 'n-core-and-length-three-rounds'],
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    @staticmethod
    def _normalize_history(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    def _load_raw_users(self) -> pd.DataFrame:
        pnt(f'loading RecIF video histories from {self.source_path}')
        users = pd.read_parquet(self.source_path, columns=[self.UID_COL, 'hist_video_pid'])
        users = users.rename(columns={'hist_video_pid': self.HIS_COL})
        users[self.HIS_COL] = users[self.HIS_COL].apply(self._normalize_history)
        users = users[users[self.HIS_COL].map(len) > 0].reset_index(drop=True)
        pnt(f'loaded {len(users)} raw RecIF video users with non-empty history')
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
            caption_pid_set.update(pid_array.tolist())
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

    def _apply_length_constraints(self, users: pd.DataFrame, desc: str) -> pd.DataFrame:
        trimmed_histories = []
        kept_uids = []
        iterator = zip(users[self.UID_COL].tolist(), users[self.HIS_COL].tolist())
        for uid, history in tqdm(iterator, total=len(users), desc=desc):
            if len(history) > self.max_length:
                history = history[-self.max_length:]
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
        pnt(f'loading captions for {len(final_item_ids)} filtered video items from {self.caption_path}')
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
            raise ValueError('No RecIF video items with captions survived filtering')
        items = pd.concat(rows, ignore_index=True)
        items = items.rename(columns={'dense_caption': 'caption'})
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
            f'RecIF video formatting complete with items={len(self._filtered_items)} '
            f'users={len(self._filtered_users)}'
        )

    def load_items(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return cast(pd.DataFrame, self._filtered_items)

    def load_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return self._filtered_users


class RecIFVideoLargeFormatter(RecIFVideoFormatter):
    DEFAULT_N_CORE = 10
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50


class RecIFVideoXLargeFormatter(RecIFVideoFormatter):
    DEFAULT_N_CORE = 3
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50


class RecIFVideoXLargeAlignFormatter(RecIFVideoXLargeFormatter):
    VER = 'v1.1'
    PROVIDES_TEST_SET = True

    @staticmethod
    def _parse_answer_pids(metadata_value):
        if isinstance(metadata_value, str):
            try:
                metadata = json.loads(metadata_value)
            except json.JSONDecodeError:
                return []
        elif isinstance(metadata_value, dict):
            metadata = metadata_value
        else:
            return []

        raw_answer = metadata.get('answer_pid')
        if raw_answer is None:
            raw_answer = metadata.get('answer_pids')
        if isinstance(raw_answer, np.ndarray):
            raw_answer = raw_answer.tolist()
        if isinstance(raw_answer, tuple):
            raw_answer = list(raw_answer)
        if not isinstance(raw_answer, list):
            return []
        return [pid for pid in raw_answer if pid is not None]

    def load_test_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        final_item_ids = set(cast(pd.DataFrame, self._filtered_items)[self.IID_COL].tolist())
        pnt(f'loading aligned RecIF video official test from {self.official_video_test_path}')
        test_users = pd.read_parquet(self.official_video_test_path, columns=['hist_pid', 'metadata'])
        test_users[self.HIS_COL] = test_users['hist_pid'].apply(self._normalize_history)
        test_users['answer_pids'] = test_users['metadata'].apply(self._parse_answer_pids)
        test_users[self.UID_COL] = [f'official_test_{index}' for index in range(len(test_users))]

        aligned_histories = []
        aligned_answers = []
        kept_uids = []
        iterator = zip(
            test_users[self.UID_COL].tolist(),
            test_users[self.HIS_COL].tolist(),
            test_users['answer_pids'].tolist(),
        )
        for uid, history, answer_pids in tqdm(iterator, total=len(test_users), desc='align-official-test'):
            filtered_history = [pid for pid in history if pid in final_item_ids]
            filtered_answers = [pid for pid in answer_pids if pid in final_item_ids]
            if len(filtered_history) > self.max_length:
                filtered_history = filtered_history[-self.max_length:]
            if len(filtered_history) < self.min_length or not filtered_answers:
                continue
            kept_uids.append(uid)
            aligned_histories.append(filtered_history)
            aligned_answers.append(filtered_answers)

        official_test = pd.DataFrame(
            {self.UID_COL: kept_uids, self.HIS_COL: aligned_histories, 'answer_pids': aligned_answers}
        )
        pnt(f'aligned official test users kept={len(official_test)} from raw={len(test_users)}')
        return official_test
