import argparse
import json
from pathlib import Path

import numpy as np
import pigmento
from pigmento import pnt
from tqdm import tqdm

from utils import get_data_dir, load_embedder, load_processor


class Embedder:
    def __init__(self, conf):
        self.conf = conf

        self.data = conf.data.lower()
        self.model_name = conf.model.replace('.', '').lower()

        data_dir = conf.data_dir or get_data_dir(self.data)
        self.processor = load_processor(self.data, data_dir=data_dir)
        self.processor.load()

        self.caller = load_embedder(
            self.model_name,
            device=self.conf.device,
            batch_size=self.conf.batch_size,
        ).post_init()

        self.cache_dir = Path('cache') / 'embeddings' / self.data / self.model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_path = self.cache_dir / 'embeddings.npy'
        self.item_ids_path = self.cache_dir / 'item_ids.parquet'
        self.meta_path = self.cache_dir / 'meta.json'

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
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))

    def is_cached(self):
        return self.embedding_path.exists() and self.item_ids_path.exists() and self.meta_path.exists()

    def embed(self):
        if self.is_cached() and not self.conf.overwrite:
            pnt(f'cached embeddings found at {self.embedding_path}')
            return

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
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False,
    )

    parser = argparse.ArgumentParser(description='Extract item embeddings from items.parquet content.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind or movielens.')
    parser.add_argument('--model', required=True, help='Embedder model name.')
    parser.add_argument('--data_dir', default=None, help='Optional raw data directory override.')
    parser.add_argument('--device', default='cpu', help='Device string, such as cpu or cuda:0.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding.')
    parser.add_argument('--normalize', action='store_true', help='Apply L2 normalization to embeddings.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cached embeddings.')
    args = parser.parse_args()

    embedder = Embedder(args)
    embedder.embed()
