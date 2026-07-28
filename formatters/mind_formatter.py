import os
from typing import cast

import pandas as pd
from pigmento import pnt

from formatters.base_formatter import BaseFormatter


class MINDFormatter(BaseFormatter):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'train', 'news.tsv')
        return pd.read_csv(
            filepath_or_buffer=path,
            sep='\t',
            names=[self.IID_COL, 'cat', 'subcat', 'title', 'abs', 'url', 'tit_ent', 'abs_ent'],
            usecols=[self.IID_COL, 'cat', 'subcat', 'title', 'abs'],
        )

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'train', 'behaviors.tsv')
        users = pd.read_csv(
            filepath_or_buffer=path,
            sep='\t',
            names=['imp', self.UID_COL, 'time', self.HIS_COL, 'predict'],
            usecols=[self.UID_COL, 'time', self.HIS_COL],
        )

        users['time'] = pd.to_datetime(users['time'], errors='coerce')
        users[self.HIS_COL] = users[self.HIS_COL].str.split()
        users = users.dropna(subset=[self.HIS_COL])
        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: [item for item in x if item in item_set])
        users = users[users[self.HIS_COL].map(lambda x: len(x) > 0)]
        return users

    def deduplicate_users(self, users: pd.DataFrame):
        users = users.copy()
        users['history_len'] = users[self.HIS_COL].map(len)
        users = users.sort_values(
            [self.UID_COL, 'time', 'history_len'],
            kind='stable',
        ).groupby(self.UID_COL, sort=False).tail(1)
        return users[[self.UID_COL, self.HIS_COL]].reset_index(drop=True)


class MINDFFormatter(MINDFormatter):
    VER = 'v1.0-full'
    MAX_HISTORY_LENGTH = 256

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self._full_items: pd.DataFrame | None = None
        self._full_users: pd.DataFrame | None = None
        self._full_stats: dict = {}

    @staticmethod
    def _split_history(value):
        if isinstance(value, str):
            return value.split()
        return []

    def _load_raw_items(self) -> pd.DataFrame:
        return super().load_items()

    def _load_raw_user_rows(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'train', 'behaviors.tsv')
        return pd.read_csv(
            filepath_or_buffer=path,
            sep='\t',
            names=['imp', self.UID_COL, 'time', self.HIS_COL, 'predict'],
            usecols=[self.UID_COL, 'time', self.HIS_COL],
        )

    def _run_full_pipeline(self):
        if self._full_items is not None and self._full_users is not None:
            return

        pnt(f'MIND full formatting settings: max_length={self.MAX_HISTORY_LENGTH}, item_n_core=disabled')

        raw_items = self._load_raw_items()
        raw_item_set = set(raw_items[self.IID_COL].dropna().unique())

        raw_users = self._load_raw_user_rows()
        raw_users[self.HIS_COL] = raw_users[self.HIS_COL].apply(self._split_history)
        raw_users['time'] = pd.to_datetime(raw_users['time'], errors='coerce')

        item_count_before = int(raw_items[self.IID_COL].dropna().nunique())
        user_count_before = int(raw_users[self.UID_COL].dropna().nunique())
        interaction_count_before = int(raw_users[self.HIS_COL].map(len).sum())
        sequence_lengths_before = self._history_length_stats(raw_users[self.HIS_COL].tolist())

        users = raw_users.copy()
        users[self.HIS_COL] = users[self.HIS_COL].apply(
            lambda history: [item for item in history if item in raw_item_set][-self.MAX_HISTORY_LENGTH :]
        )
        users = users[users[self.HIS_COL].map(len) > 0].reset_index(drop=True)
        users = self.deduplicate_users(users)

        final_item_ids = set()
        for history in users[self.HIS_COL].tolist():
            final_item_ids.update(history)

        items = raw_items[raw_items[self.IID_COL].isin(final_item_ids)]
        items = items.drop_duplicates(subset=[self.IID_COL], keep='first').reset_index(drop=True)

        interaction_count_after = int(users[self.HIS_COL].map(len).sum())
        sequence_lengths_after = self._history_length_stats(users[self.HIS_COL].tolist())
        self._full_stats = {
            'formatter_mode': 'full',
            'max_history_length_limit': int(self.MAX_HISTORY_LENGTH),
            'item_filter_policy': 'no-n-core; keep items observed in final user histories',
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

        self._full_items = items
        self._full_users = users

        pnt(
            f'MIND full count transition: items {item_count_before}->{len(items)} '
            f'users {user_count_before}->{len(users)} '
            f'interactions {interaction_count_before}->{interaction_count_after}'
        )
        self._print_history_length_stats('MIND full before', sequence_lengths_before)
        self._print_history_length_stats('MIND full after', sequence_lengths_after)
        pnt(
            f'MIND full formatting complete with items={len(self._full_items)} users={len(self._full_users)} '
            f'interactions={interaction_count_after}'
        )

    def _extra_meta(self):
        return {
            'formatter_mode': 'full',
            'max_history_length': int(self.MAX_HISTORY_LENGTH),
            'item_filter_policy': 'no-n-core',
            'history_policy': 'latest-items',
        }

    def _cache_meta_matches(self, cached_meta):
        return int(cached_meta.get('max_history_length', -1)) == int(self.MAX_HISTORY_LENGTH)

    def _extra_stats(self):
        return dict(self._full_stats)

    def load_items(self) -> pd.DataFrame:
        self._run_full_pipeline()
        return cast(pd.DataFrame, self._full_items)

    def load_users(self) -> pd.DataFrame:
        self._run_full_pipeline()
        return cast(pd.DataFrame, self._full_users)
