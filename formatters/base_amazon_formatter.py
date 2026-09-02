import gzip
import json
from pathlib import Path

import pandas as pd

from formatters.base_uict_formatter import UICTFormatter


class AmazonFormatter(UICTFormatter):
    VER = 'v1.0-amazon-unified'

    UID_COL = 'reviewerID'
    IID_COL = 'asin'
    HIS_COL = 'history'
    LBL_COL = 'click'
    DAT_COL = 'reviewTime'
    RAT_COL = 'overall'

    MAX_LINES = 0
    REQUIRE_STRINGIFY = False

    LEGACY_REVIEW_PATTERNS = (
        '{subset}.json.gz',
        '{subset}_5.json.gz',
        'reviews_{subset}.json.gz',
        'reviews_{subset}_5.json.gz',
    )
    MODERN_REVIEW_PATTERNS = (
        'raw_review_{subset}.jsonl.gz',
        '{subset}.jsonl.gz',
    )
    META_PATTERNS = (
        'meta_{subset}.json.gz',
        'meta_{subset}.jsonl.gz',
        'raw_meta_{subset}.jsonl.gz',
    )

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

    def _resolve_path(self, patterns, label):
        if not self.data_dir:
            raise ValueError(
                f'No raw data directory configured for {self.get_name()}. '
                f'Add `{self.get_name()} = /path/to/amazon-beauty` to .data.'
            )
        data_dir = Path(self.data_dir)
        candidates = [data_dir / pattern.format(subset=self.subset) for pattern in patterns]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        expected = ', '.join(path.name for path in candidates)
        raise FileNotFoundError(f'Amazon {label} file not found in {data_dir}; expected one of: {expected}')

    @staticmethod
    def _rename_first(frame, target, candidates):
        if target in frame.columns:
            return frame
        for candidate in candidates:
            if candidate in frame.columns:
                return frame.rename(columns={candidate: target})
        raise ValueError(
            f'Amazon data is missing required field {target!r}; available fields: '
            f'{", ".join(map(str, frame.columns))}'
        )

    def _normalize_items(self, items):
        items = self._rename_first(items, self.IID_COL, ('parent_asin', 'item_id'))
        items = self._rename_first(items, 'title', ('name',))
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
        interactions = self._rename_first(interactions, self.UID_COL, ('user_id',))
        interactions = self._rename_first(interactions, self.IID_COL, ('parent_asin', 'item_id'))
        interactions = self._rename_first(interactions, self.RAT_COL, ('rating',))

        if self.DAT_COL in interactions.columns:
            timestamps = pd.to_datetime(interactions[self.DAT_COL], errors='coerce')
        elif 'unixReviewTime' in interactions.columns:
            timestamps = pd.to_datetime(interactions['unixReviewTime'], unit='s', errors='coerce')
        elif 'timestamp' in interactions.columns:
            raw_timestamps = pd.to_numeric(interactions['timestamp'], errors='coerce')
            unit = 'ms' if raw_timestamps.dropna().median() > 10_000_000_000 else 's'
            timestamps = pd.to_datetime(raw_timestamps, unit=unit, errors='coerce')
        else:
            raise ValueError(
                'Amazon reviews require one of reviewTime, unixReviewTime, or timestamp; '
                f'available fields: {", ".join(map(str, interactions.columns))}'
            )

        normalized = interactions[[self.UID_COL, self.IID_COL, self.RAT_COL]].copy()
        normalized[self.DAT_COL] = timestamps
        normalized[self.UID_COL] = normalized[self.UID_COL].astype(str)
        normalized[self.IID_COL] = normalized[self.IID_COL].astype(str)
        normalized[self.RAT_COL] = pd.to_numeric(normalized[self.RAT_COL], errors='coerce')
        return normalized.dropna(
            subset=[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL]
        ).reset_index(drop=True)

    def load_items(self) -> pd.DataFrame:
        path = self._resolve_path(self.META_PATTERNS, 'metadata')
        items = self._load_data(path)
        return self._normalize_items(items)

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = self._resolve_path(
            self.LEGACY_REVIEW_PATTERNS + self.MODERN_REVIEW_PATTERNS,
            'review',
        )
        interactions = self._load_data(path)
        interactions = self._normalize_interactions(interactions)
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL].apply(lambda x: int(x >= 4))
        interactions = interactions.drop(columns=[self.RAT_COL])

        return self._load_users(interactions)
