import os

import pandas as pd

from formatters.base_uict_formatter import UICTFormatter


class YelpFormatter(UICTFormatter):
    UID_COL = 'user_id'
    IID_COL = 'business_id'
    HIS_COL = 'history'
    DAT_COL = 'date'
    LBL_COL = 'click'
    RAT_COL = 'stars'

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['name']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'yelp_academic_dataset_business.json')
        items = pd.read_json(path, lines=True)
        return items[['business_id', 'name', 'address', 'city', 'state']]

    def load_users(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'yelp_academic_dataset_review.json')
        interactions = pd.read_json(path, lines=True)
        interactions = interactions[[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL]]

        interactions[self.RAT_COL] = interactions[self.RAT_COL].astype(int)
        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL].apply(lambda x: int(x > 3))
        interactions = interactions.drop(columns=[self.RAT_COL])
        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL])

        return self._load_users(interactions)
