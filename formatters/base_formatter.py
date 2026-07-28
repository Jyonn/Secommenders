import abc
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pigmento import pnt

from utils.artifact import ArtifactStore


class BaseFormatter(abc.ABC):
    VER = 'v2.5'
    HISTORY_LENGTH_BUCKETS = (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 4),
        (5, 8),
        (9, 16),
        (17, 32),
        (33, 64),
        (65, 128),
        (129, 256),
        (257, None),
    )

    IID_COL: str
    UID_COL: str
    HIS_COL: str

    REQUIRE_STRINGIFY: bool
    PROVIDES_TEST_SET = False
    MULTI_ITEM_COL: Optional[str] = None
    USE_ALL_USERS_IN_PROCESSOR = False
    SPLIT_RATIO = 0.9

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.store = ArtifactStore(self.get_name())
        self.store_dir = str(self.store.formatted_dir())

        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.test_users: Optional[pd.DataFrame] = None
        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

    @property
    @abc.abstractmethod
    def default_attrs(self):
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Formatter', '').lower()

    @classmethod
    def get_seed(cls):
        return cls.__name__.replace('Formatter', '').lower()

    @abc.abstractmethod
    def load_items(self) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_users(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_test_users(self) -> Optional[pd.DataFrame]:
        if self.PROVIDES_TEST_SET:
            raise NotImplementedError(
                f'formatter {self.get_name()} declares PROVIDES_TEST_SET=True but does not implement load_test_users()'
            )
        return None

    def _stringify(self, df: pd.DataFrame):
        if not self.REQUIRE_STRINGIFY:
            return df
        if self.IID_COL in df.columns:
            df[self.IID_COL] = df[self.IID_COL].astype(str)
        if self.UID_COL in df.columns:
            df[self.UID_COL] = df[self.UID_COL].astype(str)
        return df

    def _paths(self):
        base_dir = Path(self.store_dir)
        return {
            'items': base_dir / 'items.parquet',
            'users': base_dir / 'users.parquet',
            'test_users': base_dir / 'test_users.parquet',
            'meta': base_dir / 'meta.json',
            'stats': base_dir / 'stats.json',
        }

    def _extra_meta(self):
        return {}

    def _extra_stats(self):
        return {}

    def _cache_meta_matches(self, cached_meta):
        return True

    @classmethod
    def _history_length_stats(cls, histories):
        lengths = [
            len(history) if hasattr(history, '__len__') and not isinstance(history, str) else 0
            for history in histories
        ]
        if not lengths:
            return {
                'summary': {
                    'count': 0,
                    'min': None,
                    'mean': None,
                    'p25': None,
                    'p50': None,
                    'p75': None,
                    'p90': None,
                    'p95': None,
                    'p99': None,
                    'max': None,
                },
                'buckets': [],
            }

        series = pd.Series(lengths, dtype='float64')
        count = int(len(series))
        summary = {
            'count': count,
            'min': int(series.min()),
            'mean': float(series.mean()),
            'p25': float(series.quantile(0.25)),
            'p50': float(series.quantile(0.50)),
            'p75': float(series.quantile(0.75)),
            'p90': float(series.quantile(0.90)),
            'p95': float(series.quantile(0.95)),
            'p99': float(series.quantile(0.99)),
            'max': int(series.max()),
        }

        buckets = []
        for lower, upper in cls.HISTORY_LENGTH_BUCKETS:
            if upper is None:
                mask = series >= lower
                label = f'{lower}+'
            elif lower == upper:
                mask = series == lower
                label = str(lower)
            else:
                mask = (series >= lower) & (series <= upper)
                label = f'{lower}-{upper}'
            bucket_count = int(mask.sum())
            buckets.append(
                {
                    'range': label,
                    'count': bucket_count,
                    'ratio': float(bucket_count / count),
                }
            )

        return {'summary': summary, 'buckets': buckets}

    @staticmethod
    def _print_history_length_stats(label: str, stats: dict):
        summary = stats.get('summary', {})
        if not summary or not summary.get('count'):
            pnt(f'{label} history lengths: empty')
            return

        pnt(
            f'{label} history lengths: count={summary["count"]} min={summary["min"]} '
            f'mean={summary["mean"]:.2f} p50={summary["p50"]:.1f} p90={summary["p90"]:.1f} '
            f'p95={summary["p95"]:.1f} p99={summary["p99"]:.1f} max={summary["max"]}'
        )
        buckets = [bucket for bucket in stats.get('buckets', []) if bucket.get('count', 0) > 0]
        if not buckets:
            pnt(f'{label} history length histogram: empty')
            return

        max_count = max(bucket['count'] for bucket in buckets)
        width = 40
        pnt(f'{label} history length histogram:')
        for bucket in buckets:
            bar_width = max(1, int(round(bucket['count'] / max_count * width)))
            bar = '#' * bar_width
            ratio = bucket.get('ratio', 0.0) * 100
            pnt(f'  {bucket["range"]:>7} | {bar:<{width}} {bucket["count"]:>8} ({ratio:5.2f}%)')

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
        meta.update(self._extra_meta())
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    def _save_stats(self):
        stats_path = self._paths()['stats']
        history_lengths = self.users[self.HIS_COL].map(len)
        stats = {
            'item_count': int(len(self.items)),
            'user_count': int(len(self.users)),
            'avg_history_length': float(history_lengths.mean()),
            'max_history_length': int(history_lengths.max()),
        }
        if self.test_users is not None:
            test_history_lengths = self.test_users[self.HIS_COL].map(len)
            stats.update(
                {
                    'test_user_count': int(len(self.test_users)),
                    'test_avg_history_length': float(test_history_lengths.mean()),
                    'test_max_history_length': int(test_history_lengths.max()),
                }
            )
        stats.update(self._extra_stats())
        stats_path.write_text(json.dumps(stats, indent=2) + '\n')

    @staticmethod
    def _has_non_empty_history(history):
        return hasattr(history, '__len__') and not isinstance(history, str) and len(history) > 0

    def _finalize_items(self, items: pd.DataFrame):
        items = items.dropna(subset=[self.IID_COL]).reset_index(drop=True)

        duplicate_count = int(items.duplicated(subset=[self.IID_COL]).sum())
        if duplicate_count:
            pnt(f'dropping {duplicate_count} duplicate items for {self.IID_COL}')
            items = items.drop_duplicates(subset=[self.IID_COL], keep='first').reset_index(drop=True)

        return items

    def deduplicate_users(self, users: pd.DataFrame):
        duplicate_count = int(users.duplicated(subset=[self.UID_COL]).sum())
        raise ValueError(
            f'formatter {self.get_name()} produced {duplicate_count} duplicate users for {self.UID_COL}; '
            f'override deduplicate_users() to define merge policy'
        )

    def _finalize_users(self, users: pd.DataFrame):
        core_columns = [self.UID_COL, self.HIS_COL]
        extra_columns = [column for column in users.columns if column not in core_columns]
        users = users.dropna(subset=[self.UID_COL, self.HIS_COL]).reset_index(drop=True)
        users = users[users[self.HIS_COL].map(self._has_non_empty_history)].reset_index(drop=True)

        duplicate_count = int(users.duplicated(subset=[self.UID_COL]).sum())
        if duplicate_count:
            pnt(f'merging {duplicate_count} duplicate users for {self.UID_COL}')
            users = self.deduplicate_users(users)

        users = users.dropna(subset=[self.UID_COL, self.HIS_COL]).reset_index(drop=True)
        users = users[users[self.HIS_COL].map(self._has_non_empty_history)].reset_index(drop=True)

        remaining_duplicates = int(users.duplicated(subset=[self.UID_COL]).sum())
        if remaining_duplicates:
            raise ValueError(
                f'formatter {self.get_name()} still has {remaining_duplicates} duplicate users '
                f'after deduplication'
            )
        return users[core_columns + extra_columns].reset_index(drop=True)

    def load(self):
        paths = self._paths()
        items_path = paths['items']
        users_path = paths['users']
        test_users_path = paths['test_users']
        meta_path = paths['meta']
        cache_updated = False
        cache_meta_valid = False
        if meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text())
                cache_meta_valid = (
                    cached_meta.get('version') == self.VER
                    and bool(cached_meta.get('provides_test_set', False)) == bool(self.PROVIDES_TEST_SET)
                    and cached_meta.get('multi_item_col') == self.MULTI_ITEM_COL
                    and bool(cached_meta.get('use_all_users_in_processor', False))
                    == bool(self.USE_ALL_USERS_IN_PROCESSOR)
                    and float(cached_meta.get('split_ratio', -1.0)) == float(self.SPLIT_RATIO)
                    and self._cache_meta_matches(cached_meta)
                )
            except json.JSONDecodeError:
                cache_meta_valid = False

        test_cache_ready = (not self.PROVIDES_TEST_SET) or test_users_path.exists()
        if items_path.exists() and users_path.exists() and test_cache_ready and cache_meta_valid:
            pnt(f'loading formatted {self.get_name()} from cache')
            raw_items = pd.read_parquet(items_path)
            raw_users = pd.read_parquet(users_path)
            raw_test_users = pd.read_parquet(test_users_path) if self.PROVIDES_TEST_SET else None

            self.items = self._finalize_items(self._stringify(raw_items))
            self.users = self._finalize_users(self._stringify(raw_users))
            if raw_test_users is not None:
                self.test_users = self._finalize_users(self._stringify(raw_test_users))
            cache_updated = not self.items.equals(raw_items) or not self.users.equals(raw_users)
            if raw_test_users is not None:
                cache_updated = cache_updated or not self.test_users.equals(raw_test_users)
        else:
            pnt(f'loading {self.get_name()} from raw data')
            self.items = self._finalize_items(self._stringify(self.load_items()))
            self.users = self._finalize_users(self._stringify(self.load_users()))
            if self.PROVIDES_TEST_SET:
                raw_test_users = self.load_test_users()
                if raw_test_users is None:
                    raise ValueError(
                        f'formatter {self.get_name()} returned no test users despite PROVIDES_TEST_SET=True'
                    )
                self.test_users = self._finalize_users(self._stringify(raw_test_users))
            cache_updated = True

        self.items = self._stringify(self.items)
        self.users = self._stringify(self.users)
        if self.REQUIRE_STRINGIFY:
            self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(lambda x: [str(item) for item in x])
        if self.test_users is not None:
            self.test_users = self._stringify(self.test_users)
            if self.REQUIRE_STRINGIFY:
                self.test_users[self.HIS_COL] = self.test_users[self.HIS_COL].apply(lambda x: [str(item) for item in x])
                if self.MULTI_ITEM_COL and self.MULTI_ITEM_COL in self.test_users.columns:
                    self.test_users[self.MULTI_ITEM_COL] = self.test_users[self.MULTI_ITEM_COL].apply(
                        lambda x: [str(item) for item in x]
                    )

        if cache_updated:
            pnt(f'writing normalized formatted cache for {self.get_name()}')
            self.items.to_parquet(items_path, index=False)
            self.users.to_parquet(users_path, index=False)
            if self.test_users is not None:
                self.test_users.to_parquet(test_users_path, index=False)
            self._save_meta()
            self._save_stats()

        self.item_vocab = dict(zip(self.items[self.IID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.UID_COL], range(len(self.users))))

        pnt(f'loaded {len(self.items)} formatted items')
        pnt(f'loaded {len(self.users)} formatted users')
        if self.test_users is not None:
            pnt(f'loaded {len(self.test_users)} formatted test users')
        return self

    def organize_item(self, iid, item_attrs: list, as_dict=False, item_self=False):
        item = iid if item_self else self.items.iloc[self.item_vocab[iid]]

        if as_dict:
            return {attr: item[attr] or '' for attr in item_attrs}
        if len(item_attrs) == 1:
            return item[item_attrs[0]]
        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])


class BaseMultiTargetFormatter(BaseFormatter, abc.ABC):
    MULTI_ITEM_COL: str
