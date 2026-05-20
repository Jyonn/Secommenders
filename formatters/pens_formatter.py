import os
from typing import cast

import pandas as pd

from formatters.base_formatter import BaseFormatter


class PENSFormatter(BaseFormatter):
    IID_COL = 'nid'
    UID_COL = 'uid'
    HIS_COL = 'history'

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'news.tsv')
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            header=0,
            names=[self.IID_COL, 'category', 'topic', 'title', 'body', 'entity', 'content'],
            usecols=[self.IID_COL, 'category', 'topic', 'title', 'body'],
        )

    def _load_user(self, mode):
        path = os.path.join(self.data_dir, f'{mode}.tsv')
        return pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep='\t',
            header=0,
            names=[self.UID_COL, 'history', 'dwell_time', 'exposure_time', 'pos', 'neg', 'start', 'end', 'dwell_time_pos'],
            usecols=[self.UID_COL, 'history', 'exposure_time'],
        )

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        users_train = self._load_user('train')
        users_dev = self._load_user('valid')
        users = pd.concat([users_train, users_dev]).reset_index(drop=True)

        users['exposure_time'] = pd.to_datetime(users['exposure_time'], errors='coerce')
        users[self.HIS_COL] = users[self.HIS_COL].str.split()
        users[self.HIS_COL] = users[self.HIS_COL].apply(lambda x: [item for item in x if item in item_set])
        users = users[users[self.HIS_COL].map(lambda x: len(x) > 0)]
        return users

    def deduplicate_users(self, users: pd.DataFrame):
        users = users.copy()
        users['history_len'] = users[self.HIS_COL].map(len)
        users = users.sort_values(
            [self.UID_COL, 'exposure_time', 'history_len'],
            kind='stable',
        ).groupby(self.UID_COL, sort=False).tail(1)
        return users[[self.UID_COL, self.HIS_COL]].reset_index(drop=True)
