from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pigmento import pnt
from tqdm import tqdm

from embedders.base_model import BaseModel
from utils.recif_embedding_cache import ensure_filtered_recif_embeddings


class RecIFPretrainModel(BaseModel):
    KEY = 'recif-pretrain'
    EMBEDDING_COLUMNS = ()
    READ_BATCH_SIZE = 100_000

    def encode(self, samples: list[str], normalize=False) -> np.ndarray:
        raise RuntimeError(f'{self.get_name()} reads RecIF provided embeddings and does not encode text.')

    @staticmethod
    def _normalize_item_id(value):
        if value is None:
            return None
        try:
            numeric = float(value)
            if numeric.is_integer():
                return int(numeric)
        except (TypeError, ValueError):
            pass
        return str(value)

    @staticmethod
    def _as_vector(value):
        if value is None:
            raise ValueError('embedding value is null')
        if hasattr(value, 'as_py'):
            value = value.as_py()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'embedding value must be a flat vector, got {type(value).__name__}')
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            raise ValueError('embedding value must already be mean-pooled before embedder loads it')

        vector = np.asarray(value, dtype=np.float32)
        if vector.size == 0:
            raise ValueError('embedding value is empty')
        return vector

    def embed_items(
        self,
        item_ids: list,
        data_dir: str | Path,
        normalize=False,
        dataset: str | None = None,
    ):
        if not self.EMBEDDING_COLUMNS:
            raise ValueError(f'{self.get_name()} does not define EMBEDDING_COLUMNS')
        if not data_dir:
            raise ValueError('RecIF pretrain embeddings require a data_dir configured in .data')

        embeddings_dir = Path(data_dir) / 'embeddings'
        if not embeddings_dir.exists():
            raise FileNotFoundError(f'RecIF pretrain embedding directory not found: {embeddings_dir}')

        normalized_item_ids = [self._normalize_item_id(item_id) for item_id in item_ids]
        item_positions = {item_id: index for index, item_id in enumerate(normalized_item_ids)}
        candidate_pids = set(normalized_item_ids)
        row_seen = set()
        found_by_column = {column: {} for column in self.EMBEDDING_COLUMNS}
        parse_errors = {column: {} for column in self.EMBEDDING_COLUMNS}
        filtered_path = ensure_filtered_recif_embeddings(
            data_dir,
            dataset=dataset,
            candidate_pids=candidate_pids,
        )
        parquet_paths = [filtered_path]

        columns = ['pid', *self.EMBEDDING_COLUMNS]
        pnt(
            f'loading RecIF pretrain embeddings columns={list(self.EMBEDDING_COLUMNS)} '
            f'items={len(item_positions)} files={len(parquet_paths)} source=filtered cache'
        )
        batch_size = max(int(self.batch_size or 0), self.READ_BATCH_SIZE)
        for path in tqdm(parquet_paths, desc='pretrain-embeddings'):
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
                frame = batch.to_pandas()
                for row in frame.to_dict('records'):
                    item_id = self._normalize_item_id(row.get('pid'))
                    if item_id not in item_positions:
                        continue
                    row_seen.add(item_id)
                    for column in self.EMBEDDING_COLUMNS:
                        if item_id in found_by_column[column]:
                            continue
                        try:
                            found_by_column[column][item_id] = self._as_vector(row[column])
                        except ValueError as exc:
                            parse_errors[column][item_id] = str(exc)
                        except Exception as exc:
                            raise ValueError(f'failed to parse {column} for pid={item_id}: {exc}') from exc
            if all(len(found_by_column[column]) == len(item_positions) for column in self.EMBEDDING_COLUMNS):
                break

        missing = [item_id for item_id in normalized_item_ids if item_id not in row_seen]
        if missing:
            preview = ', '.join(str(item_id) for item_id in missing[:10])
            raise ValueError(
                f'RecIF filtered embedding cache is missing {len(missing)}/{len(normalized_item_ids)} '
                f'processed items for dataset={dataset or "unknown"}; first missing: {preview}. '
                'Regenerate formatted artifacts with the embedding completeness filter.'
            )

        for column in self.EMBEDDING_COLUMNS:
            found = found_by_column[column]
            if not found:
                raise ValueError(f'RecIF pretrain column {column} has no valid embeddings')
            missing_column_items = [item_id for item_id in normalized_item_ids if item_id not in found]
            if missing_column_items:
                preview = ', '.join(str(item_id) for item_id in missing_column_items[:10])
                raise ValueError(
                    f'RecIF pretrain embeddings have invalid {column} values for '
                    f'{len(missing_column_items)}/{len(normalized_item_ids)} processed items; '
                    f'first invalid: {preview}. Regenerate the dataset-family filtered embedding '
                    'cache from formatter.'
                )

        merged = []
        for item_id in normalized_item_ids:
            vectors = [found_by_column[column][item_id] for column in self.EMBEDDING_COLUMNS]
            if len(vectors) == 1:
                merged.append(vectors[0])
            else:
                merged.append(np.concatenate(vectors, axis=0))

        dims = {int(vector.shape[0]) for vector in merged}
        if len(dims) != 1:
            preview = ', '.join(str(dim) for dim in sorted(dims)[:10])
            raise ValueError(f'RecIF pretrain embeddings have inconsistent dimensions: {preview}')

        self.pretrain_stats = {
            'imputed_by_column': {column: 0 for column in self.EMBEDDING_COLUMNS},
            'parse_error_count_by_column': {
                column: len(errors)
                for column, errors in parse_errors.items()
            },
        }

        embeddings = np.stack(merged, axis=0).astype(np.float32)
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings


class PretrainVisionModel(RecIFPretrainModel):
    KEY = 'recif-pretrain-vision'
    EMBEDDING_COLUMNS = ('vision_emb',)


class PretrainTextModel(RecIFPretrainModel):
    KEY = 'recif-pretrain-text'
    EMBEDDING_COLUMNS = ('text_emb',)


class PretrainMultimodalModel(RecIFPretrainModel):
    KEY = 'recif-pretrain-multimodal'
    EMBEDDING_COLUMNS = ('vision_emb', 'text_emb')
