import gzip
import json
from pathlib import Path

import pandas as pd

from formatters.base_uict_formatter import UICTFormatter


class AmazonFormatter(UICTFormatter):
    VER = 'v1.1-amazon-explicit'

    UID_COL = 'reviewerID'
    IID_COL = 'asin'
    HIS_COL = 'history'
    LBL_COL = 'click'
    DAT_COL = 'reviewTime'
    RAT_COL = 'overall'

    MAX_LINES = 0
    REQUIRE_STRINGIFY = True

    @property
    def default_attrs(self):
        return ['title']

    def __init__(self, subset, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset

    @staticmethod
    def parse(path):
        with gzip.open(path, 'rt', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @classmethod
    def _load_data(cls, path):
        data = []
        for index, row in enumerate(cls.parse(path)):
            data.append(row)
            if cls.MAX_LINES and len(data) >= cls.MAX_LINES:
                break
        return pd.DataFrame(data)

    def _resolve_path(self, filename, label):
        if not self.data_dir:
            raise ValueError(
                f'No raw data directory configured for {self.get_name()}. '
                f'Add `{self.get_name()} = /path/to/data` to .data.'
            )
        path = Path(self.data_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f'Amazon {label} file not found: {path}')
        return path

    def _normalize_items(self, items):
        required = {self.IID_COL, 'title'}
        missing = sorted(required - set(items.columns))
        if missing:
            raise ValueError(f'Amazon metadata is missing required fields: {", ".join(missing)}')
        items = items[[self.IID_COL, 'title']].dropna(subset=[self.IID_COL, 'title'])
        items[self.IID_COL] = items[self.IID_COL].astype(str)
        items['title'] = items['title'].astype(str).str.strip()
        items = items[items['title'] != '']
        items['title'] = items['title'].str.replace(r'&#[0-9]+;', '', regex=True)
        items['title'] = items['title'].str.replace(r'&[a-zA-Z]+;', '', regex=True)
        items['title'] = items['title'].str.replace(r'[^\w\s]', '', regex=True)
        items = items[items['title'].str.strip() != '']
        return items.drop_duplicates(subset=self.IID_COL).reset_index(drop=True)

    def _normalize_interactions(self, interactions):
        required = {self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL}
        missing = sorted(required - set(interactions.columns))
        if missing:
            raise ValueError(f'Amazon reviews are missing required fields: {", ".join(missing)}')
        normalized = interactions[[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL]].copy()
        normalized[self.DAT_COL] = pd.to_datetime(normalized[self.DAT_COL], errors='coerce')
        normalized[self.UID_COL] = normalized[self.UID_COL].astype(str)
        normalized[self.IID_COL] = normalized[self.IID_COL].astype(str)
        normalized[self.RAT_COL] = pd.to_numeric(normalized[self.RAT_COL], errors='coerce')
        return normalized.dropna(
            subset=[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL]
        ).reset_index(drop=True)

    def load_items(self) -> pd.DataFrame:
        path = self._resolve_path(f'meta_{self.subset}.json.gz', 'metadata')
        items = self._load_data(path)
        return self._normalize_items(items)

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = self._resolve_path(f'{self.subset}.json.gz', 'review')
        interactions = self._load_data(path)
        interactions = self._normalize_interactions(interactions)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL].apply(lambda x: int(x >= 4))
        interactions = interactions.drop(columns=[self.RAT_COL])

        return self._load_users(interactions)
