import abc

import pandas as pd
from pigmento import pnt

from formatters.mind_formatter import MINDFormatter
from utils.stable_random import stable_shuffle


class MINDScaleFormatter(MINDFormatter, abc.ABC):
    """Build nested MIND training scales with one shared held-out test set."""

    VER = 'v1.0-mind-small-scale'

    PROVIDES_TEST_SET = True
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9

    SCALE_TEST_RATIO = 0.003
    SCALE_SHUFFLE_SEED = 'MIND'
    SCALE_SHUFFLE_VERSION = 'v1'

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self._scaled_test_users: pd.DataFrame | None = None
        self._scale_total_users: int | None = None
        self._scale_test_start: int | None = None
        self._scale_train_limit: int | None = None

    @classmethod
    @abc.abstractmethod
    def scale_percent(cls) -> int:
        raise NotImplementedError

    @classmethod
    def get_seed(cls):
        return cls.SCALE_SHUFFLE_SEED

    def _scale_split_points(self, total_users: int):
        test_start = int(total_users * (1.0 - self.SCALE_TEST_RATIO))
        if total_users > 1:
            test_start = min(max(test_start, 1), total_users - 1)
        train_limit = int(total_users * (self.scale_percent() / 100.0))
        train_limit = min(train_limit, test_start)
        return test_start, train_limit

    def _extra_meta(self):
        meta = super()._extra_meta()
        meta.update(
            {
                'source_dataset': 'mind',
                'scale_percent': int(self.scale_percent()),
                'scale_test_ratio': float(self.SCALE_TEST_RATIO),
                'scale_shuffle_seed': self.get_seed(),
                'scale_shuffle_version': self.SCALE_SHUFFLE_VERSION,
                'scale_split_policy': 'deduplicate-users-stable-shuffle-prefix-train-tail-test',
            }
        )
        if self._scale_total_users is not None:
            meta.update(
                {
                    'scale_total_users': int(self._scale_total_users),
                    'scale_test_start': int(self._scale_test_start),
                    'scale_train_limit': int(self._scale_train_limit),
                }
            )
        return meta

    def _cache_meta_matches(self, cached_meta):
        return (
            super()._cache_meta_matches(cached_meta)
            and cached_meta.get('source_dataset') == 'mind'
            and int(cached_meta.get('scale_percent', -1)) == int(self.scale_percent())
            and float(cached_meta.get('scale_test_ratio', -1.0)) == float(self.SCALE_TEST_RATIO)
            and cached_meta.get('scale_shuffle_seed') == self.get_seed()
            and cached_meta.get('scale_shuffle_version') == self.SCALE_SHUFFLE_VERSION
            and cached_meta.get('scale_total_users') is not None
            and cached_meta.get('scale_test_start') is not None
            and cached_meta.get('scale_train_limit') is not None
        )

    def load_users(self) -> pd.DataFrame:
        raw_users = super().load_users()
        users = self.deduplicate_users(raw_users)
        records = stable_shuffle(users.to_dict('records'), seed=self.get_seed())

        test_start, train_limit = self._scale_split_points(len(records))
        if train_limit <= 0:
            raise ValueError(
                f'scale={self.scale_percent()}% produced no train users from {len(records)} deduplicated users'
            )

        columns = list(users.columns)
        train_users = pd.DataFrame(records[:train_limit], columns=columns).reset_index(drop=True)
        test_users = pd.DataFrame(records[test_start:], columns=columns).reset_index(drop=True)
        if test_users.empty:
            raise ValueError(f'scale split produced no test users from {len(records)} deduplicated users')

        self._scaled_test_users = test_users
        self._scale_total_users = len(records)
        self._scale_test_start = test_start
        self._scale_train_limit = train_limit

        pnt(
            f'MIND small-scale formatting complete scale={self.scale_percent()}% '
            f'train_users={len(train_users)} test_users={len(test_users)} total_users={len(records)}'
        )
        return train_users

    def load_test_users(self) -> pd.DataFrame:
        if self._scaled_test_users is None:
            raise RuntimeError('load_users() must run before load_test_users()')
        return self._scaled_test_users


class MINDS1Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 1


class MINDS2Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 2


class MINDS5Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 5


class MINDS10Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 10


class MINDS20Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 20


class MINDS30Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 30


class MINDS40Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 40


class MINDS50Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 50


class MINDS60Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 60


class MINDS70Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 70


class MINDS80Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 80


class MINDS90Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 90


class MINDS95Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 95


class MINDS99Formatter(MINDScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 99
