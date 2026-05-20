import os
from typing import cast

import pandas as pd

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
