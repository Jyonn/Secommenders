import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from utils import get_data_dir, load_embedder, load_processor
from utils.artifact import ArtifactStore
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

        self.caller = load_embedder(
            self.model_name,
            device=self.device,
            batch_size=self.conf.batch_size,
        ).post_init()

        self.embedding_dir = ArtifactStore(self.data).embedded_dir(self.model_name)

        self.embedding_path = self.embedding_dir / 'embeddings.npy'
        self.item_ids_path = self.embedding_dir / 'item_ids.parquet'
        self.meta_path = self.embedding_dir / 'meta.json'

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
        }
        if hasattr(self.caller, 'embed_items'):
            meta['source'] = 'recif-pretrain-parquet'
            meta['data_dir'] = self.data_dir
            meta['embedding_columns'] = list(getattr(self.caller, 'EMBEDDING_COLUMNS', ()))
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

    def embed(self):
        if self.is_cached() and not self.conf.overwrite:
            pnt(f'cached embeddings found at {self.embedding_path}')
            return

        if hasattr(self.caller, 'embed_items'):
            pnt(f'loading RecIF provided embeddings for {self.data}/{self.model_name}')
            item_ids = self.processor.items[self.processor.IID_COL].tolist()
            embeddings = self.caller.embed_items(item_ids, data_dir=self.data_dir, normalize=self.conf.normalize)
            np.save(self.embedding_path, embeddings)
            self.processor.items[[self.processor.IID_COL]].to_parquet(self.item_ids_path, index=False)
            self.save_meta(embeddings)
            pnt(f'embeddings saved to {self.embedding_path}')
            return

        pnt(f'loading item content from {self.items_path}')
        contents = self.get_contents()
        embeddings = []
        total = len(contents)
        num_batches = (total + self.caller.batch_size - 1) // self.caller.batch_size
        pnt(f'encoding {total} items on {self.data} with {self.model_name}')

        for batch in tqdm(self.caller.iter_batches(contents), total=num_batches):
            batch_embeddings = self.caller.encode(batch, normalize=self.conf.normalize)
            embeddings.append(batch_embeddings)

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
