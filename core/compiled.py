import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import function
from utils.artifact import ArtifactStore
from utils.pipeline import ensure_compiled


class CompiledArtifacts:
    def __init__(self, config):
        self.config = config
        self.store = ArtifactStore(config.data)
        self.compile_dir = self.store.compiled_dir(config.compile_config.prepare_id)
        self.meta = None
        self.prompt_main = None
        self.prompt_align = None
        self.vocab_meta = None
        self.special_vocab = None
        self.uid_raw_items = None
        self.item_views = {}
        self.finetune = None
        self.test = None
        self.embedding_matrix = None
        self.sid_num_quantizers = None
        self.sid_codebook_size = None

    def _read_json(self, path: Path):
        return json.loads(path.read_text())

    def _load_view(self, name: str):
        path = self.compile_dir / 'item_views' / f'{name}.parquet'
        if not path.exists():
            return None
        values = pd.read_parquet(path)['value'].tolist()
        return [function.to_list(value) for value in values]

    def _load_embedding_matrix(self):
        if 'embedding' not in self.item_views:
            return None
        if not self.config.repr_model:
            raise ValueError('repr.model is required when compiled data uses embedding views')
        embedding_dir = self.store.embedded_dir(self.config.repr_model)
        embedding_path = embedding_dir / 'embeddings.npy'
        if not embedding_path.exists():
            raise FileNotFoundError(f'Embedding matrix not found: {embedding_path}')
        matrix = np.load(embedding_path).astype(np.float32)
        return torch.tensor(matrix, dtype=torch.float32)

    def load(self):
        required_paths = [
            self.compile_dir / 'meta.json',
            self.compile_dir / 'samples' / 'finetune.parquet',
            self.compile_dir / 'samples' / 'test.parquet',
            self.compile_dir / 'vocab' / 'uid.json',
            self.compile_dir / 'vocab' / 'special.json',
            self.compile_dir / 'vocab' / 'meta.json',
            self.compile_dir / 'prompts' / 'main.json',
            self.compile_dir / 'prompts' / 'alignment.json',
            self.compile_dir / 'item_views' / 'uid.parquet',
        ]
        if not all(path.exists() for path in required_paths):
            ensure_compiled(self.config.compile_config)

        self.meta = self._read_json(self.compile_dir / 'meta.json')
        self.vocab_meta = self._read_json(self.compile_dir / 'vocab' / 'meta.json')
        self.special_vocab = self._read_json(self.compile_dir / 'vocab' / 'special.json')
        self.prompt_main = self._read_json(self.compile_dir / 'prompts' / 'main.json')
        self.prompt_align = self._read_json(self.compile_dir / 'prompts' / 'alignment.json')
        self.uid_raw_items = self._read_json(self.compile_dir / 'vocab' / 'uid.json')['raw_item_ids']
        self.finetune = pd.read_parquet(self.compile_dir / 'samples' / 'finetune.parquet')
        self.test = pd.read_parquet(self.compile_dir / 'samples' / 'test.parquet')

        for view_name in ['uid', 'text', 'sid', 'embedding']:
            values = self._load_view(view_name)
            if values is not None:
                self.item_views[view_name] = values

        sid_vocab_path = self.compile_dir / 'vocab' / 'sid.json'
        if sid_vocab_path.exists():
            sid_vocab = self._read_json(sid_vocab_path)
            self.sid_num_quantizers = int(sid_vocab['num_quantizers'])
            self.sid_codebook_size = int(sid_vocab['codebook_size'])

        self.embedding_matrix = self._load_embedding_matrix()
        return self

    @property
    def num_items(self):
        return len(self.uid_raw_items)

    @property
    def model_vocab_size(self):
        namespace = next(entry for entry in self.vocab_meta['namespaces'] if entry['kind'] == 'model')
        return int(namespace['size'])

    @property
    def sid_vocab_size(self):
        sid_entries = [entry for entry in self.vocab_meta['namespaces'] if entry['kind'] == 'sid']
        return int(sid_entries[0]['size']) if sid_entries else 0

    @property
    def model_kind(self):
        return self.meta['model_kind']

    @property
    def model_key(self):
        model_vocab = self._read_json(self.compile_dir / 'vocab' / 'model.json')
        return model_vocab.get('model_key')
