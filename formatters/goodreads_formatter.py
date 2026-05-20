import json
import os
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from formatters.base_uict_formatter import UICTFormatter


class GoodreadsFormatter(UICTFormatter):
    IID_COL = 'bid'
    UID_COL = 'uid'
    HIS_COL = 'history'
    DAT_COL = 'date'
    LBL_COL = 'click'

    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'goodreads_book_works.json')
        items = pd.read_json(path, lines=True)
        items = items[['best_book_id', 'original_title']]
        items = items[items['original_title'].str.strip() != '']
        items.columns = [self.IID_COL, 'title']
        return items

    @staticmethod
    def _str_to_ts(date_string):
        date_format = '%a %b %d %H:%M:%S %z %Y'
        dt = datetime.strptime(date_string, date_format)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, 'goodreads_interactions_dedup.json')
        interactions = []
        with open(path, 'r') as file:
            for index, line in tqdm(enumerate(file)):
                if index > 1e7:
                    break
                data = json.loads(line.strip())
                interactions.append([data['user_id'], data['book_id'], data['is_read'], data['date_added']])

        interactions = pd.DataFrame(interactions, columns=[self.UID_COL, self.IID_COL, self.LBL_COL, self.DAT_COL])
        interactions = self._stringify(interactions)
        interactions[self.DAT_COL] = interactions[self.DAT_COL].apply(self._str_to_ts)
        interactions[self.LBL_COL] = interactions[self.LBL_COL].apply(int)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        return self._load_users(interactions)
