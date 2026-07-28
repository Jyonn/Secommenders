import pandas as pd

from formatters.recif_base import RecIFBaseFormatter, RecIFFullFormatterMixin


class RecIFVideoFormatter(RecIFBaseFormatter):
    DOMAIN = 'video'
    RAW_HISTORY_COL = 'hist_video_pid'
    OFFICIAL_TEST_DIR = 'video'
    OFFICIAL_TEST_HISTORY_COL = 'hist_pid'


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
        return self._load_official_test_users()


class RVFFormatter(RecIFFullFormatterMixin, RecIFVideoFormatter):
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9

    @classmethod
    def get_seed(cls):
        return 'rvf'
