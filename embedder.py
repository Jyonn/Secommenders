import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from utils import get_data_dir, load_embedder, load_processor
from utils.artifact import ArtifactStore
from utils.artifact_run import ArtifactRunCoordinator
from utils.gpu import GPU
from utils.logging import setup_logging


class Embedder:
    def __init__(self, conf):
        self.conf = conf

        self.data = conf.data.lower()
        self.model_name = conf.model.replace('.', '').lower()
        self.device = conf.device or GPU.auto_choose(torch_format=True)

        data_dir = get_data_dir(self.data)
        self.data_dir = data_dir
        self.processor = load_processor(self.data, data_dir=data_dir)
        self.processor.load()
        self.items_path = Path(self.processor.store_dir) / 'items.parquet'

        self.caller = None

        self.embedding_dir = ArtifactStore(self.data).embedded_dir(self.model_name)

        self.embedding_path = self.embedding_dir / 'embeddings.npy'
        self.item_ids_path = self.embedding_dir / 'item_ids.parquet'
        self.meta_path = self.embedding_dir / 'meta.json'
        self.run_state = None

    def _ensure_caller(self):
        if self.caller is None:
            self.caller = load_embedder(
                self.model_name,
                device=self.device,
                batch_size=self.conf.batch_size,
            ).post_init()
        return self.caller

    def get_contents(self):
        contents = []
        for item_id in self.processor.items[self.processor.IID_COL]:
            content = self.processor.organize_item(item_id, item_attrs=self.processor.default_attrs)
            contents.append(content or '[Empty Content]')
        return contents

    def save_meta(self, embeddings):
        meta = {
            'dataset': self.data,
            'model': self.model_name,
            'model_key': self.caller.key,
            'item_count': int(len(self.processor.items)),
            'embedding_dim': int(embeddings.shape[1]),
            'content_attrs': list(self.processor.default_attrs),
            'processed_items_path': str(self.items_path),
            'normalize': bool(self.conf.normalize),
            'status': 'completed',
        }
        if hasattr(self.caller, 'embed_items'):
            meta['source'] = 'recif-pretrain-parquet'
            meta['data_dir'] = self.data_dir
            meta['embedding_columns'] = list(getattr(self.caller, 'EMBEDDING_COLUMNS', ()))
            meta['pretrain_stats'] = getattr(self.caller, 'pretrain_stats', {})
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def save_subset_reuse_meta(self, embeddings, source_dataset: str, source_dir: Path, source_meta: dict):
        meta = {
            'dataset': self.data,
            'model': self.model_name,
            'model_key': self.caller.key,
            'item_count': int(len(self.processor.items)),
            'embedding_dim': int(embeddings.shape[1]),
            'content_attrs': list(self.processor.default_attrs),
            'processed_items_path': str(self.items_path),
            'normalize': bool(self.conf.normalize),
            'status': 'completed',
            'reuse_mode': 'scale-subset',
            'source_dataset': source_dataset,
            'source_embedding_dir': str(source_dir),
            'source_item_count': int(source_meta.get('item_count', embeddings.shape[0])),
        }
        if hasattr(self.caller, 'embed_items'):
            meta['source'] = source_meta.get('source', 'recif-pretrain-parquet')
            meta['data_dir'] = self.data_dir
            meta['embedding_columns'] = list(getattr(self.caller, 'EMBEDDING_COLUMNS', ()))
            meta['source_pretrain_stats'] = source_meta.get('pretrain_stats', {})
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def is_cached(self):
        if not (self.embedding_path.exists() and self.item_ids_path.exists() and self.meta_path.exists()):
            return False
        try:
            meta = json.loads(self.meta_path.read_text())
        except json.JSONDecodeError:
            pnt(f'invalid embedding meta found at {self.meta_path}, rebuilding cache')
            return False

        if meta.get('processed_items_path') != str(self.items_path):
            pnt('embedding cache points to a different processed items path, rebuilding cache')
            return False
        if int(meta.get('item_count', -1)) != int(len(self.processor.items)):
            pnt('embedding cache item_count mismatches current processed items, rebuilding cache')
            return False

        cached_item_ids = self._load_cached_item_ids()
        current_item_ids = self.processor.items[self.processor.IID_COL].tolist()
        if cached_item_ids != current_item_ids:
            pnt('embedding cache item id ordering mismatches current processed items, rebuilding cache')
            return False
        return True

    def _load_cached_item_ids(self):
        frame = pd.read_parquet(self.item_ids_path)
        frame = self.processor._stringify(frame)
        return frame[self.processor.IID_COL].tolist()

    @staticmethod
    def _scale_dataset_info(dataset: str):
        dataset = str(dataset).lower()
        prefix = dataset[:2]
        suffix = dataset[2:]
        if prefix not in {'rv', 'ra'} or not suffix.isdigit():
            return None
        return prefix, int(suffix)

    def _iter_larger_scale_embedding_candidates(self):
        target = self._scale_dataset_info(self.data)
        if target is None:
            return []
        prefix, target_percent = target
        embedded_root = ArtifactStore.ROOT / 'embedded'
        candidates = []
        for dataset_dir in embedded_root.glob(f'{prefix}*'):
            if not dataset_dir.is_dir():
                continue
            source = self._scale_dataset_info(dataset_dir.name)
            if source is None:
                continue
            _, source_percent = source
            if source_percent <= target_percent:
                continue
            embedding_dir = dataset_dir / self.model_name
            candidates.append((source_percent, dataset_dir.name.lower(), embedding_dir))
        return sorted(candidates, key=lambda item: item[0])

    def _load_item_ids_from_path(self, path: Path):
        frame = pd.read_parquet(path)
        if self.processor.IID_COL in frame.columns:
            column = self.processor.IID_COL
        else:
            column = frame.columns[0]
        return frame[column].tolist()

    def _embedding_meta_compatible(self, meta: dict, embeddings: np.ndarray):
        if meta.get('model') != self.model_name:
            return False
        if meta.get('model_key') != self.caller.key:
            return False
        if bool(meta.get('normalize', False)) != bool(self.conf.normalize):
            return False
        if int(meta.get('item_count', -1)) != int(embeddings.shape[0]):
            return False
        if int(meta.get('embedding_dim', -1)) != int(embeddings.shape[1]):
            return False
        if hasattr(self.caller, 'embed_items'):
            expected_columns = list(getattr(self.caller, 'EMBEDDING_COLUMNS', ()))
            if list(meta.get('embedding_columns', [])) != expected_columns:
                return False
        return True

    def try_reuse_larger_scale_embeddings(self):
        target_item_ids = self.processor.items[self.processor.IID_COL].tolist()
        target_keys = [str(item_id) for item_id in target_item_ids]
        for _, source_dataset, source_dir in self._iter_larger_scale_embedding_candidates():
            embedding_path = source_dir / 'embeddings.npy'
            item_ids_path = source_dir / 'item_ids.parquet'
            meta_path = source_dir / 'meta.json'
            if not (embedding_path.exists() and item_ids_path.exists() and meta_path.exists()):
                continue
            try:
                source_meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            source_embeddings = np.load(embedding_path, mmap_mode='r')
            if not self._embedding_meta_compatible(source_meta, source_embeddings):
                continue
            source_item_ids = self._load_item_ids_from_path(item_ids_path)
            if len(source_item_ids) != source_embeddings.shape[0]:
                continue
            positions = {}
            for index, item_id in enumerate(source_item_ids):
                positions.setdefault(str(item_id), index)
            missing = [key for key in target_keys if key not in positions]
            if missing:
                pnt(
                    f'scale embedding candidate {source_dataset}/{self.model_name} misses '
                    f'{len(missing)}/{len(target_keys)} target items; rebuilding'
                )
                continue
            gather_indices = np.asarray([positions[key] for key in target_keys], dtype=np.int64)
            embeddings = np.lib.format.open_memmap(
                self.embedding_path,
                mode='w+',
                dtype=np.float32,
                shape=(len(gather_indices), int(source_embeddings.shape[1])),
            )
            chunk_size = 100_000
            for start in range(0, len(gather_indices), chunk_size):
                end = min(start + chunk_size, len(gather_indices))
                embeddings[start:end] = np.asarray(source_embeddings[gather_indices[start:end]], dtype=np.float32)
            embeddings.flush()
            self.processor.items[[self.processor.IID_COL]].to_parquet(self.item_ids_path, index=False)
            self.save_subset_reuse_meta(embeddings, source_dataset, source_dir, source_meta)
            pnt(
                f'reused embeddings from {source_dataset}/{self.model_name} for {self.data}/{self.model_name} '
                f'items={len(target_item_ids)}'
            )
            return True
        return False

    def embed(self):
        cache_was_ready = self.is_cached()
        self.run_state = ArtifactRunCoordinator(
            self.embedding_dir,
            kind='embedder',
            identity=f'{self.data}/{self.model_name}',
        )
        if not self.run_state.acquire_or_wait(self.is_cached, force_producer=bool(self.conf.overwrite)):
            if cache_was_ready and not self.conf.overwrite:
                pnt(f'cached embeddings found at {self.embedding_path}')
            else:
                pnt(f'using embeddings prepared by another process at {self.embedding_path}')
            return

        try:
            self._embed_as_owner()
            if not self.is_cached():
                raise RuntimeError(f'embedder finished without a valid cache at {self.embedding_dir}')
            self.run_state.finish(message=f'embeddings ready at {self.embedding_path}')
        except BaseException as exc:
            self.run_state.fail(exc)
            raise

    def _embed_as_owner(self):
        self.run_state.update(stage='initializing-model', message=f'loading {self.model_name}')
        self._ensure_caller()

        if self.try_reuse_larger_scale_embeddings():
            return

        if hasattr(self.caller, 'embed_items'):
            self.run_state.update(
                stage='loading-precomputed',
                current=0,
                total=len(self.processor.items),
                message='loading provided item embeddings',
            )
            pnt(f'loading RecIF provided embeddings for {self.data}/{self.model_name}')
            item_ids = self.processor.items[self.processor.IID_COL].tolist()
            embeddings = self.caller.embed_items(
                item_ids,
                data_dir=self.data_dir,
                normalize=self.conf.normalize,
                dataset=self.data,
            )
            np.save(self.embedding_path, embeddings)
            self.processor.items[[self.processor.IID_COL]].to_parquet(self.item_ids_path, index=False)
            self.save_meta(embeddings)
            self.run_state.update(
                stage='saving',
                current=len(item_ids),
                total=len(item_ids),
                message='precomputed embeddings saved',
            )
            pnt(f'embeddings saved to {self.embedding_path}')
            return

        pnt(f'loading item content from {self.items_path}')
        contents = self.get_contents()
        embeddings = []
        total = len(contents)
        num_batches = (total + self.caller.batch_size - 1) // self.caller.batch_size
        pnt(f'encoding {total} items on {self.data} with {self.model_name}')

        for batch_index, batch in enumerate(tqdm(self.caller.iter_batches(contents), total=num_batches), start=1):
            batch_embeddings = self.caller.encode(batch, normalize=self.conf.normalize)
            embeddings.append(batch_embeddings)
            self.run_state.update(
                stage='encoding',
                current=batch_index,
                total=num_batches,
                message=f'encoded batch {batch_index}/{num_batches}',
            )

        self.run_state.update(stage='saving', message='writing embedding artifacts')
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        np.save(self.embedding_path, embeddings)
        self.processor.items[[self.processor.IID_COL]].to_parquet(self.item_ids_path, index=False)
        self.save_meta(embeddings)
        pnt(f'embeddings saved to {self.embedding_path}')


if __name__ == '__main__':
    setup_logging()

    parser = argparse.ArgumentParser(description='Extract item embeddings from items.parquet content.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    parser.add_argument('--model', required=True, help='Embedder model name.')
    parser.add_argument('--device', default=None, help='Device string, such as cpu or cuda:0.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding.')
    parser.add_argument('--normalize', action='store_true', help='Apply L2 normalization to embeddings.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cached embeddings.')
    args = parser.parse_args()

    embedder = Embedder(args)
    embedder.embed()
