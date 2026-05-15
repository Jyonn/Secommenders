import os

import pandas as pd
from tqdm import tqdm

from processors.base_processor import BaseProcessor


class HMProcessor(BaseProcessor):
    IID_COL = 'article_id'
    UID_COL = 'customer_id'
    HIS_COL = 'history'
    DAT_COL = 't_dat'

    NUM_TEST = 5_000
    NUM_FINETUNE = 40_000
    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['detail_desc']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'articles.csv')
        article = pd.read_csv(filepath_or_buffer=path, sep=',', dtype={self.IID_COL: str})

        article = article[[self.IID_COL, 'detail_desc']]
        article = article[article['detail_desc'].notnull()]
        article = article.drop_duplicates(subset=self.IID_COL).reset_index(drop=True)
        return article

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())
        path = os.path.join(self.data_dir, 'transactions_train.csv')

        interactions = []
        with open(path, 'r') as file:
            file.readline()
            for index, line in tqdm(enumerate(file)):
                if index > 1e7:
                    break
                interactions.append(line.strip().split(',')[:3])

        interactions = pd.DataFrame(interactions, columns=[self.DAT_COL, self.UID_COL, self.IID_COL])
        interactions = self._stringify(interactions)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        users = interactions.sort_values(
            [self.UID_COL, self.DAT_COL]
        ).groupby(self.UID_COL)[self.IID_COL].apply(list).reset_index()
        users.columns = [self.UID_COL, self.HIS_COL]
        return users
