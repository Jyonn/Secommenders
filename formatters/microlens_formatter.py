import os

import pandas as pd

from formatters.base_formatter import BaseFormatter


class MicroLensFormatter(BaseFormatter):
    IID_COL = 'item'
    UID_COL = 'user'
    HIS_COL = 'history'
    DAT_COL = 'timestamp'

    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'MicroLens-50k_titles.csv')
        titles = pd.read_csv(filepath_or_buffer=path, sep=',')
        return titles[[self.IID_COL, 'title']]

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'MicroLens-50k_pairs.csv')
        interactions = pd.read_csv(filepath_or_buffer=path, sep=',')
        interactions = self._stringify(interactions)

        users = interactions.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]
        return users
