from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pigmento import pnt
from tqdm import tqdm

from embedders.base_model import BaseModel


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

        if isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 1:
            vectors = []
            for item in value.tolist():
                try:
                    vectors.append(RecIFPretrainModel._as_vector(item))
                except ValueError as exc:
                    if 'null' in str(exc) or 'empty' in str(exc):
                        continue
                    raise
            if not vectors:
                raise ValueError('embedding value is empty')
            dims = {int(vector.shape[0]) for vector in vectors}
            if len(dims) == 1 and len(vectors) > 1:
                return np.stack(vectors, axis=0).mean(axis=0).astype(np.float32)
            if len(vectors) == 1:
                return vectors[0]

        if isinstance(value, (list, tuple)) and value:
            vectors = []
            for item in value:
                try:
                    vectors.append(RecIFPretrainModel._as_vector(item))
                except ValueError as exc:
                    if 'null' in str(exc) or 'empty' in str(exc):
                        continue
                    raise
            if not vectors:
                raise ValueError('embedding value is empty')
            dims = {int(vector.shape[0]) for vector in vectors}
            if len(dims) == 1 and len(vectors) > 1:
                return np.stack(vectors, axis=0).mean(axis=0).astype(np.float32)
            if len(vectors) == 1:
                return vectors[0]

        def flatten(item):
            if hasattr(item, 'as_py'):
                item = item.as_py()
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                for child in item:
                    yield from flatten(child)
                return
            if item is None:
                return
            yield float(item)

        vector = np.fromiter(flatten(value), dtype=np.float32)
        if vector.size == 0:
            raise ValueError('embedding value is empty')
        return vector

    def _merge_vectors(self, row: dict):
        vectors = []
        item_id = row.get('pid')
        for column in self.EMBEDDING_COLUMNS:
            try:
                vectors.append(self._as_vector(row[column]))
            except Exception as exc:
                raise ValueError(f'failed to parse {column} for pid={item_id}: {exc}') from exc
        if len(vectors) == 1:
            return vectors[0]
        return np.concatenate(vectors, axis=0)

    @staticmethod
    def _mean_vector(vectors, *, column: str):
        dims = {int(vector.shape[0]) for vector in vectors}
        if len(dims) != 1:
            preview = ', '.join(str(dim) for dim in sorted(dims)[:10])
            raise ValueError(f'RecIF pretrain {column} embeddings have inconsistent dimensions: {preview}')
        return np.stack(list(vectors), axis=0).mean(axis=0).astype(np.float32)

    def embed_items(self, item_ids: list, data_dir: str | Path, normalize=False):
        if not self.EMBEDDING_COLUMNS:
            raise ValueError(f'{self.get_name()} does not define EMBEDDING_COLUMNS')
        if not data_dir:
            raise ValueError('RecIF pretrain embeddings require a data_dir configured in .data')

        embeddings_dir = Path(data_dir) / 'embeddings'
        if not embeddings_dir.exists():
            raise FileNotFoundError(f'RecIF pretrain embedding directory not found: {embeddings_dir}')

        normalized_item_ids = [self._normalize_item_id(item_id) for item_id in item_ids]
        item_positions = {item_id: index for index, item_id in enumerate(normalized_item_ids)}
        row_seen = set()
        found_by_column = {column: {} for column in self.EMBEDDING_COLUMNS}
        parse_errors = {column: {} for column in self.EMBEDDING_COLUMNS}
        parquet_paths = sorted(embeddings_dir.glob('*.parquet'))
        if not parquet_paths:
            raise FileNotFoundError(f'No parquet files found in RecIF pretrain embedding directory: {embeddings_dir}')

        columns = ['pid', *self.EMBEDDING_COLUMNS]
        pnt(
            f'loading RecIF pretrain embeddings columns={list(self.EMBEDDING_COLUMNS)} '
            f'items={len(item_positions)} files={len(parquet_paths)}'
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
                f'RecIF pretrain embedding rows missing {len(missing)}/{len(normalized_item_ids)} processed items; '
                f'first missing: {preview}'
            )

        imputed_by_column = {}
        for column in self.EMBEDDING_COLUMNS:
            found = found_by_column[column]
            if not found:
                raise ValueError(f'RecIF pretrain column {column} has no valid embeddings')
            fallback = self._mean_vector(found.values(), column=column)
            missing_column_items = [item_id for item_id in normalized_item_ids if item_id not in found]
            for item_id in missing_column_items:
                found[item_id] = fallback
            imputed_by_column[column] = len(missing_column_items)
            if missing_column_items:
                preview = ', '.join(str(item_id) for item_id in missing_column_items[:10])
                pnt(
                    f'imputed {len(missing_column_items)} missing {column} embeddings with global mean; '
                    f'first missing: {preview}'
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
            'imputed_by_column': imputed_by_column,
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
