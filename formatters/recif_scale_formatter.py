import abc
import json
from pathlib import Path
from typing import cast

import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from formatters.recif_base import RecIFBaseFormatter
from utils.recif_embedding_cache import filtered_embeddings_path, load_filtered_embedding_pids
from utils.stable_random import stable_shuffle


class RecIFScaleFormatter(RecIFBaseFormatter, abc.ABC):
    VER = 'v1.3-scale'

    PROVIDES_TEST_SET = True
    USE_ALL_USERS_IN_PROCESSOR = True
    SPLIT_RATIO = 0.9

    SCALE_TEST_RATIO = 0.003
    SCALE_SHUFFLE_SEED = 'RECIF'
    SCALE_SHUFFLE_VERSION = 'v1'

    DEFAULT_N_CORE = 20
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 50

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self._filtered_test_users: pd.DataFrame | None = None
        self._scale_total_sequences: int | None = None
        self._scale_test_start: int | None = None
        self._scale_train_limit: int | None = None
        self._scale_reused_from: str | None = None

    @classmethod
    @abc.abstractmethod
    def scale_percent(cls) -> int:
        raise NotImplementedError

    @classmethod
    def get_seed(cls):
        return cls.SCALE_SHUFFLE_SEED

    def _extra_meta(self):
        meta = {
            'scale_percent': int(self.scale_percent()),
            'scale_test_ratio': float(self.SCALE_TEST_RATIO),
            'scale_shuffle_seed': self.get_seed(),
            'scale_shuffle_version': self.SCALE_SHUFFLE_VERSION,
            'scale_split_policy': 'stable-shuffle-prefix-train-tail-test',
            'filter_pipeline': [
                'caption-exists',
                'repeat-three-rounds:item-n-core-and-non-overlap-max-length-chunks',
                'complete-pretrain-embedding-filter',
                'stable-shuffle-sequences',
                'prefix-scale-train-tail-test',
            ],
            'embedding_filter_cache': str(filtered_embeddings_path(self.data_dir)),
            'embedding_filter_policy': 'drop items without complete text and vision embeddings',
        }
        if self._scale_total_sequences is not None:
            meta.update(
                {
                    'scale_total_sequences': int(self._scale_total_sequences),
                    'scale_test_start': int(self._scale_test_start),
                    'scale_train_limit': int(self._scale_train_limit),
                }
            )
        return meta

    def _cache_meta_matches(self, cached_meta):
        return (
            super()._cache_meta_matches(cached_meta)
            and int(cached_meta.get('scale_percent', -1)) == int(self.scale_percent())
            and float(cached_meta.get('scale_test_ratio', -1.0)) == float(self.SCALE_TEST_RATIO)
            and cached_meta.get('scale_shuffle_seed') == self.get_seed()
            and cached_meta.get('scale_shuffle_version') == self.SCALE_SHUFFLE_VERSION
            and cached_meta.get('scale_total_sequences') is not None
            and cached_meta.get('scale_test_start') is not None
            and cached_meta.get('scale_train_limit') is not None
        )

    @classmethod
    def _scale_dataset_prefix(cls):
        return 'ra' if cls.DOMAIN == 'ad' else 'rv'

    @classmethod
    def _parse_scale_dataset(cls, dataset: str):
        prefix = cls._scale_dataset_prefix()
        dataset = str(dataset).lower()
        if not dataset.startswith(prefix):
            return None
        suffix = dataset[len(prefix):]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _scale_split_points(self, total_sequences: int):
        test_start = int(total_sequences * (1.0 - self.SCALE_TEST_RATIO))
        if total_sequences > 1:
            test_start = min(max(test_start, 1), total_sequences - 1)
        train_limit = int(total_sequences * (self.scale_percent() / 100.0))
        train_limit = min(train_limit, test_start)
        return test_start, train_limit

    def _scale_meta_compatible(self, meta: dict):
        return (
            meta.get('version') == self.VER
            and meta.get('domain') == self.DOMAIN
            and meta.get('raw_history_col') == self.RAW_HISTORY_COL
            and int(meta.get('n_core', -1)) == int(self.n_core)
            and int(meta.get('min_length', -1)) == int(self.min_length)
            and int(meta.get('max_length', -1)) == int(self.max_length)
            and int(meta.get('filter_rounds', -1)) == int(self.FILTER_ROUNDS)
            and float(meta.get('scale_test_ratio', -1.0)) == float(self.SCALE_TEST_RATIO)
            and meta.get('scale_shuffle_seed') == self.get_seed()
            and meta.get('scale_shuffle_version') == self.SCALE_SHUFFLE_VERSION
            and meta.get('scale_total_sequences') is not None
            and meta.get('scale_test_start') is not None
        )

    @staticmethod
    def _safe_int(value):
        if value is None or pd.isna(value):
            return None
        return int(value)

    @staticmethod
    def _safe_str(value):
        if value is None or pd.isna(value):
            return None
        return str(value)

    def _iter_larger_scale_candidates(self):
        target_percent = int(self.scale_percent())
        formatted_root = Path(self.store_dir).parent
        prefix = self._scale_dataset_prefix()
        candidates = []
        for meta_path in formatted_root.glob(f'{prefix}*/meta.json'):
            dataset = meta_path.parent.name.lower()
            if dataset == self.get_name():
                continue
            scale_percent = self._parse_scale_dataset(dataset)
            if scale_percent is None or scale_percent <= target_percent:
                continue
            candidates.append((scale_percent, dataset, meta_path))
        return sorted(candidates, key=lambda item: item[0])

    def _records_to_users_from_frame(self, frame: pd.DataFrame, split: str):
        rows = []
        for index, row in enumerate(frame.to_dict('records')):
            history = self._normalize_history_ids(row[self.HIS_COL])
            source_uid = self._safe_str(row.get('source_uid'))
            if source_uid is None:
                source_uid = self._safe_str(row.get(self.UID_COL))
            output_row = {
                self.UID_COL: f'{self.get_name()}-{split}-{index:08d}',
                self.HIS_COL: history,
                'source_uid': source_uid,
                'segment_index': index,
                'segment_length': len(history),
            }
            if 'global_sequence_index' in row:
                global_sequence_index = self._safe_int(row['global_sequence_index'])
                if global_sequence_index is not None:
                    output_row['global_sequence_index'] = global_sequence_index
            rows.append(output_row)
        return pd.DataFrame(rows)

    def _try_load_from_larger_scale(self):
        for source_percent, dataset, meta_path in self._iter_larger_scale_candidates():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            if not self._scale_meta_compatible(meta):
                continue
            if int(meta.get('scale_percent', -1)) != int(source_percent):
                continue

            source_dir = meta_path.parent
            items_path = source_dir / 'items.parquet'
            users_path = source_dir / 'users.parquet'
            test_users_path = source_dir / 'test_users.parquet'
            if not (items_path.exists() and users_path.exists() and test_users_path.exists()):
                continue

            total_sequences = int(meta['scale_total_sequences'])
            test_start, train_limit = self._scale_split_points(total_sequences)
            if train_limit <= 0:
                continue

            users = pd.read_parquet(users_path)
            test_users = pd.read_parquet(test_users_path)
            if len(users) < train_limit:
                continue

            source_train_limit = int(meta.get('scale_train_limit', len(users)))
            source_test_start = int(meta.get('scale_test_start'))
            if (
                source_train_limit < train_limit
                or source_test_start != test_start
                or len(test_users) != total_sequences - test_start
            ):
                continue

            self._filtered_items = pd.read_parquet(items_path).reset_index(drop=True)
            self._filtered_users = self._records_to_users_from_frame(users.iloc[:train_limit], split='train')
            self._filtered_test_users = self._records_to_users_from_frame(test_users, split='test')
            self._scale_total_sequences = total_sequences
            self._scale_test_start = test_start
            self._scale_train_limit = train_limit
            self._scale_reused_from = dataset
            pnt(
                f'RecIF {self.DOMAIN} scale reused formatted {dataset} ({source_percent}%) '
                f'for {self.get_name()} ({self.scale_percent()}%) train_sequences={train_limit} '
                f'test_sequences={len(self._filtered_test_users)} total_sequences={total_sequences}'
            )
            return True
        return False

    def _split_history_chunks(self, history: list):
        for start in range(0, len(history), self.max_length):
            chunk = history[start:start + self.max_length]
            if len(chunk) >= self.min_length:
                yield chunk

    def _users_to_sequence_records(self, users: pd.DataFrame):
        records = []
        iterator = zip(users[self.UID_COL].tolist(), users[self.HIS_COL].tolist())
        for uid, history in tqdm(iterator, total=len(users), desc='seed-sequences'):
            normalized = self._normalize_history_ids(history)
            for chunk in self._split_history_chunks(normalized):
                records.append({'source_uid': uid, self.HIS_COL: chunk})
        return records

    def _filter_and_chunk_records(self, records: list[dict], allowed_items: set, desc: str):
        filtered_records = []
        for record in tqdm(records, total=len(records), desc=desc):
            history = [pid for pid in record[self.HIS_COL] if pid in allowed_items]
            for chunk in self._split_history_chunks(history):
                filtered_records.append({'source_uid': record['source_uid'], self.HIS_COL: chunk})
        return filtered_records

    def _records_item_set(self, records: list[dict]) -> set:
        item_set = set()
        for record in records:
            item_set.update(record[self.HIS_COL])
        return item_set

    def _load_complete_embedding_item_set(self, candidate_item_ids: set) -> set:
        raw_pids = load_filtered_embedding_pids(self.data_dir, candidate_pids=candidate_item_ids)
        return {self._normalize_item_id(pid) for pid in raw_pids}

    def _records_to_users(self, records: list[dict], split: str):
        rows = []
        for index, record in enumerate(records):
            row = {
                self.UID_COL: f'{self.get_name()}-{split}-{index:08d}',
                self.HIS_COL: record[self.HIS_COL],
                'source_uid': str(record['source_uid']),
                'segment_index': index,
                'segment_length': len(record[self.HIS_COL]),
            }
            if 'global_sequence_index' in record:
                global_sequence_index = self._safe_int(record['global_sequence_index'])
                if global_sequence_index is not None:
                    row['global_sequence_index'] = global_sequence_index
            rows.append(row)
        return pd.DataFrame(rows)

    def _run_filter_pipeline(self):
        if (
            self._filtered_items is not None
            and self._filtered_users is not None
            and self._filtered_test_users is not None
        ):
            return

        pnt(
            f'RecIF {self.DOMAIN} scale formatting settings: '
            f'scale={self.scale_percent()}%, n_core={self.n_core}, '
            f'min_length={self.min_length}, max_length={self.max_length}, rounds={self.FILTER_ROUNDS}, '
            f'test_ratio={self.SCALE_TEST_RATIO:g}'
        )

        if self._try_load_from_larger_scale():
            return

        users = self._load_raw_users()
        records = self._users_to_sequence_records(users)

        caption_pid_set = self._stream_caption_pid_set()
        records = self._filter_and_chunk_records(records, caption_pid_set, desc='caption-filter-sequences')
        if not records:
            raise ValueError(f'No RecIF {self.DOMAIN} scale sequences survived caption filtering')

        for round_index in range(self.FILTER_ROUNDS):
            counts = self._count_items(record[self.HIS_COL] for record in records)
            allowed = {pid for pid, count in counts.items() if count >= self.n_core}
            pnt(
                f'scale round {round_index + 1}/{self.FILTER_ROUNDS}: '
                f'{len(allowed)} items survive n_core>={self.n_core}'
            )
            records = self._filter_and_chunk_records(
                records,
                allowed,
                desc=f'ncore-sequence-filter@{round_index + 1}',
            )
            if not records:
                raise ValueError(
                    f'No RecIF {self.DOMAIN} scale sequences survived round {round_index + 1}; '
                    'try a smaller n_core or min_length'
                )

        candidate_item_ids = self._records_item_set(records)
        before_items = len(candidate_item_ids)
        embedding_item_set = self._load_complete_embedding_item_set(candidate_item_ids)
        records = self._filter_and_chunk_records(
            records,
            embedding_item_set,
            desc='embedding-complete-filter-sequences',
        )
        if not records:
            raise ValueError(
                f'No RecIF {self.DOMAIN} scale sequences survived complete embedding filtering; '
                'check RecIF embeddings/filtered.parquet'
            )
        after_items = len(self._records_item_set(records))
        pnt(
            f'embedding completeness filter kept {after_items}/{before_items} items '
            f'and {len(records)} sequences with min_length>={self.min_length}'
        )

        final_item_ids = self._records_item_set(records)
        items = self._load_caption_rows(final_item_ids)
        final_item_set = set(items[self.IID_COL].tolist())
        records = self._filter_and_chunk_records(records, final_item_set, desc='final-caption-align-sequences')
        if not records:
            raise ValueError(f'No RecIF {self.DOMAIN} scale sequences survived final item alignment')

        records = stable_shuffle(records, seed=self.get_seed())
        for index, record in enumerate(records):
            record['global_sequence_index'] = index
        test_start, train_limit = self._scale_split_points(len(records))
        if train_limit <= 0:
            raise ValueError(f'scale={self.scale_percent()}% produced no train sequences from {len(records)} records')

        train_records = records[:train_limit]
        test_records = records[test_start:]
        if not test_records:
            raise ValueError(f'scale split produced no test sequences from {len(records)} records')

        self._filtered_items = items.sort_values(self.IID_COL).reset_index(drop=True)
        self._filtered_users = self._records_to_users(train_records, split='train')
        self._filtered_test_users = self._records_to_users(test_records, split='test')
        self._scale_total_sequences = len(records)
        self._scale_test_start = test_start
        self._scale_train_limit = train_limit

        pnt(
            f'RecIF {self.DOMAIN} scale formatting complete with items={len(self._filtered_items)} '
            f'train_sequences={len(self._filtered_users)} test_sequences={len(self._filtered_test_users)} '
            f'total_sequences={len(records)}'
        )

    def load_test_users(self) -> pd.DataFrame:
        self._run_filter_pipeline()
        return cast(pd.DataFrame, self._filtered_test_users)


class RecIFVideoScaleFormatter(RecIFScaleFormatter, abc.ABC):
    DOMAIN = 'video'
    RAW_HISTORY_COL = 'hist_video_pid'
    OFFICIAL_TEST_DIR = 'video'
    OFFICIAL_TEST_HISTORY_COL = 'hist_pid'


class RecIFSmallScaleFormatter(RecIFScaleFormatter, abc.ABC):
    FILTER_ROUNDS = 0

    def _extra_meta(self):
        meta = super()._extra_meta()
        meta.update(
            {
                'small_scale_policy': 'one-latest-sequence-per-user-with-valid-item-backfill',
                'filter_rounds_policy': 'until-stable' if int(self.FILTER_ROUNDS) <= 0 else 'fixed',
                'filter_pipeline': [
                    'caption-exists',
                    'latest-max-length-one-sequence-per-user',
                    'repeat-until-stable:item-n-core-with-valid-item-backfill',
                    'complete-pretrain-embedding-filter',
                    'repeat-until-stable:item-n-core-with-valid-item-backfill',
                    'stable-shuffle-sequences',
                    'prefix-scale-train-tail-test',
                ],
            }
        )
        return meta

    def _latest_valid_record(self, uid, raw_history: list, allowed_items: set):
        selected = []
        for pid in reversed(raw_history):
            if pid not in allowed_items:
                continue
            selected.append(pid)
            if len(selected) >= self.max_length:
                break
        if len(selected) < self.min_length:
            return None
        selected.reverse()
        return {'source_uid': uid, self.HIS_COL: selected}

    def _build_latest_records(self, raw_users: pd.DataFrame, allowed_items: set, desc: str):
        records = []
        iterator = zip(raw_users[self.UID_COL].tolist(), raw_users[self.HIS_COL].tolist())
        for uid, raw_history in tqdm(iterator, total=len(raw_users), desc=desc):
            record = self._latest_valid_record(uid, raw_history, allowed_items)
            if record is not None:
                records.append(record)
        return records

    def _stabilize_latest_records(self, raw_users: pd.DataFrame, allowed_items: set, stage: str):
        current_allowed = set(allowed_items)
        max_rounds = None if int(self.FILTER_ROUNDS) <= 0 else int(self.FILTER_ROUNDS)
        round_index = 0
        records = []

        while True:
            records = self._build_latest_records(
                raw_users,
                current_allowed,
                desc=f'{stage}-latest-sequences@{round_index + 1}',
            )
            if not records:
                raise ValueError(
                    f'No RecIF {self.DOMAIN} small-scale sequences survived {stage} round {round_index + 1}; '
                    'try a smaller n_core or min_length'
                )

            counts = self._count_items(record[self.HIS_COL] for record in records)
            next_allowed = {pid for pid, count in counts.items() if count >= self.n_core}
            pnt(
                f'{stage} small round {round_index + 1}: '
                f'{len(next_allowed)} items survive n_core>={self.n_core} '
                f'from {len(records)} one-sequence users'
            )

            round_index += 1
            if next_allowed == current_allowed:
                return records, current_allowed, round_index

            current_allowed = next_allowed
            if max_rounds is not None and round_index >= max_rounds:
                records = self._build_latest_records(
                    raw_users,
                    current_allowed,
                    desc=f'{stage}-latest-sequences-final',
                )
                if not records:
                    raise ValueError(
                        f'No RecIF {self.DOMAIN} small-scale sequences survived {stage} final rebuild; '
                        'try a smaller n_core or min_length'
                    )
                return records, current_allowed, round_index

    def _run_filter_pipeline(self):
        if (
            self._filtered_items is not None
            and self._filtered_users is not None
            and self._filtered_test_users is not None
        ):
            return

        pnt(
            f'RecIF {self.DOMAIN} small-scale formatting settings: '
            f'scale={self.scale_percent()}%, n_core={self.n_core}, '
            f'min_length={self.min_length}, max_length={self.max_length}, rounds={self.FILTER_ROUNDS}, '
            f'test_ratio={self.SCALE_TEST_RATIO:g}'
        )

        if self._try_load_from_larger_scale():
            return

        raw_users = self._load_raw_users()
        caption_pid_set = self._stream_caption_pid_set()
        records, allowed_items, ncore_rounds = self._stabilize_latest_records(
            raw_users,
            caption_pid_set,
            stage='caption-ncore',
        )

        candidate_item_ids = self._records_item_set(records)
        before_items = len(candidate_item_ids)
        embedding_item_set = self._load_complete_embedding_item_set(candidate_item_ids)
        allowed_items = allowed_items & embedding_item_set
        records, allowed_items, embedding_rounds = self._stabilize_latest_records(
            raw_users,
            allowed_items,
            stage='embedding-ncore',
        )
        after_items = len(self._records_item_set(records))
        pnt(
            f'embedding completeness filter kept {after_items}/{before_items} items '
            f'after backfill and ncore restabilization rounds={embedding_rounds}'
        )

        final_item_ids = self._records_item_set(records)
        items = self._load_caption_rows(final_item_ids)
        final_item_set = set(items[self.IID_COL].tolist())
        records = self._build_latest_records(raw_users, final_item_set, desc='final-caption-align-latest-sequences')
        if not records:
            raise ValueError(f'No RecIF {self.DOMAIN} small-scale sequences survived final item alignment')

        records = stable_shuffle(records, seed=self.get_seed())
        for index, record in enumerate(records):
            record['global_sequence_index'] = index
        test_start, train_limit = self._scale_split_points(len(records))
        if train_limit <= 0:
            raise ValueError(f'scale={self.scale_percent()}% produced no train sequences from {len(records)} records')

        train_records = records[:train_limit]
        test_records = records[test_start:]
        if not test_records:
            raise ValueError(f'scale split produced no test sequences from {len(records)} records')

        self._filtered_items = items.sort_values(self.IID_COL).reset_index(drop=True)
        self._filtered_users = self._records_to_users(train_records, split='train')
        self._filtered_test_users = self._records_to_users(test_records, split='test')
        self._scale_total_sequences = len(records)
        self._scale_test_start = test_start
        self._scale_train_limit = train_limit

        pnt(
            f'RecIF {self.DOMAIN} small-scale formatting complete with items={len(self._filtered_items)} '
            f'train_sequences={len(self._filtered_users)} test_sequences={len(self._filtered_test_users)} '
            f'total_sequences={len(records)} ncore_rounds={ncore_rounds} embedding_rounds={embedding_rounds}'
        )


class RecIFVideoSmallScaleFormatter(RecIFSmallScaleFormatter, RecIFVideoScaleFormatter, abc.ABC):
    VER = 'v1.0-video-small-scale'

    @classmethod
    def _scale_dataset_prefix(cls):
        return 'rvs'


class RecIFVideoTinyScaleFormatter(RecIFVideoSmallScaleFormatter, abc.ABC):
    VER = 'v1.0-video-tiny-scale'
    DEFAULT_N_CORE = 30
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_MAX_LENGTH = 20

    @classmethod
    def _scale_dataset_prefix(cls):
        return 'rvt'


class RecIFAdsScaleFormatter(RecIFScaleFormatter, abc.ABC):
    DOMAIN = 'ad'
    RAW_HISTORY_COL = 'hist_ad_pid'
    OFFICIAL_TEST_DIR = 'ad'
    OFFICIAL_TEST_HISTORY_COL = 'hist_ad'

    def _normalize_item_id(self, value):
        if pd.isna(value):
            return value
        return int(value)


class RecIFAdsSmallScaleFormatter(RecIFSmallScaleFormatter, RecIFAdsScaleFormatter, abc.ABC):
    VER = 'v1.0-ads-small-scale'

    @classmethod
    def _scale_dataset_prefix(cls):
        return 'ras'


class RVS1Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 1


class RVS2Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 2


class RVS5Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 5


class RVS10Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 10


class RVS20Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 20


class RVS30Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 30


class RVS40Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 40


class RVS50Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 50


class RVS60Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 60


class RVS70Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 70


class RVS80Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 80


class RVS90Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 90


class RVS95Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 95


class RVS99Formatter(RecIFVideoSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 99


class RVT1Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 1


class RVT2Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 2


class RVT5Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 5


class RVT10Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 10


class RVT20Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 20


class RVT30Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 30


class RVT40Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 40


class RVT50Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 50


class RVT60Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 60


class RVT70Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 70


class RVT80Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 80


class RVT90Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 90


class RVT95Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 95


class RVT99Formatter(RecIFVideoTinyScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 99


class RAS1Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 1


class RAS2Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 2


class RAS5Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 5


class RAS10Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 10


class RAS20Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 20


class RAS30Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 30


class RAS40Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 40


class RAS50Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 50


class RAS60Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 60


class RAS70Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 70


class RAS80Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 80


class RAS90Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 90


class RAS95Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 95


class RAS99Formatter(RecIFAdsSmallScaleFormatter):
    @classmethod
    def scale_percent(cls) -> int:
        return 99
