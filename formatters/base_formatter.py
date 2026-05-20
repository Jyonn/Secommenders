import abc
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pigmento import pnt

from utils.artifact import ArtifactStore


class BaseFormatter(abc.ABC):
    VER = 'v2.1'
    REGISTER = True

    IID_COL: str
    UID_COL: str
    HIS_COL: str

    REQUIRE_STRINGIFY: bool

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.store = ArtifactStore(self.get_name())
        self.store_dir = str(self.store.formatted_dir())

        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.item_vocab: Optional[dict] = None
        self.user_vocab: Optional[dict] = None

    @property
    @abc.abstractmethod
    def default_attrs(self):
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.__name__.replace('Formatter', '').lower()

    @abc.abstractmethod
    def load_items(self) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_users(self) -> pd.DataFrame:
        raise NotImplementedError

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
        return (
            base_dir / 'items.parquet',
            base_dir / 'users.parquet',
            base_dir / 'meta.json',
            base_dir / 'stats.json',
        )

    def _save_meta(self):
        _, _, meta_path, _ = self._paths()
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
        }
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    def _save_stats(self):
        _, _, _, stats_path = self._paths()
        history_lengths = self.users[self.HIS_COL].map(len)
        stats = {
            'item_count': int(len(self.items)),
            'user_count': int(len(self.users)),
            'avg_history_length': float(history_lengths.mean()),
            'max_history_length': int(history_lengths.max()),
        }
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
        return users[[self.UID_COL, self.HIS_COL]].reset_index(drop=True)

    def load(self):
        items_path, users_path, _, _ = self._paths()
        cache_updated = False

        if items_path.exists() and users_path.exists():
            pnt(f'loading formatted {self.get_name()} from cache')
            raw_items = pd.read_parquet(items_path)
            raw_users = pd.read_parquet(users_path)

            self.items = self._finalize_items(self._stringify(raw_items))
            self.users = self._finalize_users(self._stringify(raw_users))
            cache_updated = not self.items.equals(raw_items) or not self.users.equals(raw_users)
        else:
            pnt(f'loading {self.get_name()} from raw data')
            self.items = self._finalize_items(self._stringify(self.load_items()))
            self.users = self._finalize_users(self._stringify(self.load_users()))
            cache_updated = True

        self.items = self._stringify(self.items)
        self.users = self._stringify(self.users)
        if self.REQUIRE_STRINGIFY:
            self.users[self.HIS_COL] = self.users[self.HIS_COL].apply(lambda x: [str(item) for item in x])

        if cache_updated:
            pnt(f'writing normalized formatted cache for {self.get_name()}')
            self.items.to_parquet(items_path, index=False)
            self.users.to_parquet(users_path, index=False)
            self._save_meta()
            self._save_stats()

        self.item_vocab = dict(zip(self.items[self.IID_COL], range(len(self.items))))
        self.user_vocab = dict(zip(self.users[self.UID_COL], range(len(self.users))))

        pnt(f'loaded {len(self.items)} formatted items')
        pnt(f'loaded {len(self.users)} formatted users')
        return self

    def organize_item(self, iid, item_attrs: list, as_dict=False, item_self=False):
        item = iid if item_self else self.items.iloc[self.item_vocab[iid]]

        if as_dict:
            return {attr: item[attr] or '' for attr in item_attrs}
        if len(item_attrs) == 1:
            return item[item_attrs[0]]
        return ', '.join([f'{attr}: {item[attr]}' for attr in item_attrs])
