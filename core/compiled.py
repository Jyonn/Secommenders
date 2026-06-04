import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from pigmento import pnt

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
        self.vocab_meta = None
        self.special_vocab = None
        self.uid_raw_items = None
        self.item_views = {}
        self.finetune = None
        self.valid = None
        self.test = None
        self.embedding_matrix = None
        self.sid_num_quantizers = None
        self.sid_base_num_quantizers = None
        self.sid_codebook_size = None
        self.sid_collision_vocab_size = None
        self.sid_collision_token_offset = None
        self.sid_quantizer_name = None
        self.sid_quantizer_scheme = None
        self.sid_recommended_decoding = None
        self.sid_prefix_to_next = {}
        self.sid_sequence_to_items = {}
        self.hash_num_tokens = None
        self.hash_base_num_tokens = None
        self.hash_codebook_size = None
        self.hash_collision_vocab_size = None
        self.hash_collision_token_offset = None
        self.hash_quantizer_name = None
        self.hash_quantizer_scheme = None
        self.hash_recommended_decoding = None
        self.hash_sequence_to_items = {}

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
        if not self.config.repr_source_model:
            raise ValueError('data.repr_source_model is required when compiled data uses embedding views')
        embedding_dir = self.store.embedded_dir(self.config.repr_source_model)
        embedding_path = embedding_dir / 'embeddings.npy'
        if not embedding_path.exists():
            raise FileNotFoundError(f'Embedding matrix not found: {embedding_path}')
        matrix = np.load(embedding_path).astype(np.float32)
        return torch.tensor(matrix, dtype=torch.float32)

    def _ensure_required_paths(self, required_paths: list[Path]):
        if all(path.exists() for path in required_paths):
            return

        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if distributed:
            rank = dist.get_rank()
            if rank == 0:
                pnt(f'compiled artifacts missing, rank0 is preparing {self.compile_dir}')
                ensure_compiled(self.config.compile_config)
            dist.barrier()
        else:
            ensure_compiled(self.config.compile_config)

        if not all(path.exists() for path in required_paths):
            missing = [str(path) for path in required_paths if not path.exists()]
            raise FileNotFoundError(f'Compiled artifacts still missing after auto preparation: {missing[:3]}')

    def load(self):
        required_paths = [
            self.compile_dir / 'meta.json',
            self.compile_dir / 'samples' / 'finetune.parquet',
            self.compile_dir / 'samples' / 'valid.parquet',
            self.compile_dir / 'samples' / 'test.parquet',
            self.compile_dir / 'vocab' / 'uid.json',
            self.compile_dir / 'vocab' / 'special.json',
            self.compile_dir / 'vocab' / 'meta.json',
            self.compile_dir / 'prompts' / 'main.json',
            self.compile_dir / 'item_views' / 'uid.parquet',
        ]
        self._ensure_required_paths(required_paths)

        self.meta = self._read_json(self.compile_dir / 'meta.json')
        self.vocab_meta = self._read_json(self.compile_dir / 'vocab' / 'meta.json')
        self.special_vocab = self._read_json(self.compile_dir / 'vocab' / 'special.json')
        self.prompt_main = self._read_json(self.compile_dir / 'prompts' / 'main.json')
        self.uid_raw_items = self._read_json(self.compile_dir / 'vocab' / 'uid.json')['raw_item_ids']
        self.finetune = pd.read_parquet(self.compile_dir / 'samples' / 'finetune.parquet')
        self.valid = pd.read_parquet(self.compile_dir / 'samples' / 'valid.parquet')
        self.test = pd.read_parquet(self.compile_dir / 'samples' / 'test.parquet')

        for view_name in ['uid', 'text', 'sid', 'hash', 'embedding']:
            values = self._load_view(view_name)
            if values is not None:
                self.item_views[view_name] = values

        sid_vocab_path = self.compile_dir / 'vocab' / 'sid.json'
        if sid_vocab_path.exists():
            sid_vocab = self._read_json(sid_vocab_path)
            self.sid_num_quantizers = int(sid_vocab['num_quantizers'])
            self.sid_base_num_quantizers = int(sid_vocab.get('base_num_quantizers', self.sid_num_quantizers))
            self.sid_codebook_size = int(sid_vocab['codebook_size'])
            self.sid_collision_vocab_size = int(sid_vocab.get('collision_vocab_size', 0))
            self.sid_collision_token_offset = int(sid_vocab.get('collision_token_offset', 0))
            self.sid_quantizer_name = sid_vocab.get('quantizer_name')
            self.sid_quantizer_scheme = sid_vocab.get('quantizer_scheme')
            self.sid_recommended_decoding = sid_vocab.get('recommended_decoding')
            self._build_sid_indices()

        hash_vocab_path = self.compile_dir / 'vocab' / 'hash.json'
        if hash_vocab_path.exists():
            hash_vocab = self._read_json(hash_vocab_path)
            self.hash_num_tokens = int(hash_vocab['num_tokens'])
            self.hash_base_num_tokens = int(hash_vocab.get('base_num_tokens', self.hash_num_tokens))
            self.hash_codebook_size = int(hash_vocab['codebook_size'])
            self.hash_collision_vocab_size = int(hash_vocab.get('collision_vocab_size', 0))
            self.hash_collision_token_offset = int(hash_vocab.get('collision_token_offset', 0))
            self.hash_quantizer_name = hash_vocab.get('quantizer_name')
            self.hash_quantizer_scheme = hash_vocab.get('quantizer_scheme')
            self.hash_recommended_decoding = hash_vocab.get('recommended_decoding')
            self._build_hash_indices()

        self.embedding_matrix = self._load_embedding_matrix()
        return self

    def _build_sid_indices(self):
        sid_view = self.item_views.get('sid')
        if sid_view is None:
            self.sid_prefix_to_next = {}
            self.sid_sequence_to_items = {}
            return

        prefix_to_next = {}
        sequence_to_items = {}
        for uid, codes in enumerate(sid_view):
            sequence = tuple(int(code) for code in function.to_list(codes))
            if not sequence:
                continue
            sequence_to_items.setdefault(sequence, []).append(uid)
            for prefix_len in range(len(sequence)):
                prefix = sequence[:prefix_len]
                prefix_to_next.setdefault(prefix, set()).add(sequence[prefix_len])

        self.sid_prefix_to_next = {
            prefix: sorted(next_codes)
            for prefix, next_codes in prefix_to_next.items()
        }
        self.sid_sequence_to_items = sequence_to_items

    def _build_hash_indices(self):
        hash_view = self.item_views.get('hash')
        if hash_view is None:
            self.hash_sequence_to_items = {}
            return

        sequence_to_items = {}
        for uid, codes in enumerate(hash_view):
            sequence = tuple(int(code) for code in function.to_list(codes))
            if not sequence:
                continue
            sequence_to_items.setdefault(sequence, []).append(uid)
        self.hash_sequence_to_items = sequence_to_items

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
    def hash_vocab_size(self):
        hash_entries = [entry for entry in self.vocab_meta['namespaces'] if entry['kind'] == 'hash']
        return int(hash_entries[0]['size']) if hash_entries else 0

    @property
    def model_kind(self):
        return self.meta['model_kind']

    @property
    def model_key(self):
        model_vocab = self._read_json(self.compile_dir / 'vocab' / 'model.json')
        return model_vocab.get('model_key')
