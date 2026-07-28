import abc

import pandas as pd

from formatters.recif_base import RecIFBaseFormatter, RecIFFullFormatterMixin


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


class RAFFormatter(RecIFFullFormatterMixin, RecIFAdsFormatter):
    @classmethod
    def get_seed(cls):
        return 'raf'
