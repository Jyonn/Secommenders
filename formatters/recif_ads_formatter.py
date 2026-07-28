import abc
from typing import cast

import pandas as pd
from pigmento import pnt

from formatters.recif_base import RecIFBaseFormatter


class RecIFAdsFormatter(RecIFBaseFormatter, abc.ABC):
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
        return self._load_official_test_users()


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


class RAFFormatter(RecIFAdsFormatter):
    VER = 'v1.0-full'
    FILTER_ROUNDS = 0
    DEFAULT_N_CORE = 1
    DEFAULT_MIN_LENGTH = 1
    DEFAULT_MAX_LENGTH = 256

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self._full_stats: dict = {}

    @classmethod
    def get_seed(cls):
        return 'raf'

    def _extra_meta(self):
        return {
            'formatter_mode': 'full',
            'max_history_length': int(self.max_length),
            'item_filter_policy': 'no-n-core',
            'history_policy': 'latest-items',
            'filter_pipeline': ['caption-exists-once', 'latest-256', 'drop-empty'],
        }

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
            f'max_length={self.max_length}, item_n_core=disabled'
        )

        raw_users = self._load_raw_users()
        raw_histories = raw_users[self.HIS_COL].tolist()
        item_count_before = int(len(self._history_item_set(raw_users)))
        user_count_before = int(raw_users[self.UID_COL].dropna().nunique())
        interaction_count_before = int(sum(len(history) for history in raw_histories))

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

        interaction_count_after = int(users[self.HIS_COL].map(len).sum())
        self._full_stats = {
            'formatter_mode': 'full',
            'max_history_length_limit': int(self.max_length),
            'item_filter_policy': 'no-n-core; keep items observed in final user histories',
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
        }

        self._filtered_items = items.sort_values(self.IID_COL).reset_index(drop=True)
        self._filtered_users = users.reset_index(drop=True)

        pnt(
            f'RecIF {self.DOMAIN} full formatting complete with items={len(self._filtered_items)} '
            f'users={len(self._filtered_users)} interactions={interaction_count_after}'
        )

    def load_items(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return cast(pd.DataFrame, self._filtered_items)

    def load_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return cast(pd.DataFrame, self._filtered_users)
