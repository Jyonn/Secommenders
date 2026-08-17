import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from pigmento import pnt

from utils import function
from utils.artifact import ArtifactStore
from utils.artifact_identity import resolve_compiled_dir
from utils.pipeline import ensure_compiled


class CompiledArtifacts:
    def __init__(self, config):
        self.config = config
        self.store = ArtifactStore(config.data)
        self.compile_dir = resolve_compiled_dir(config.compile_config)
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
        self.embedding_matrices = {}
        self.representation_types = {}
        self.sid_metadata = {}
        self.sid_prefix_to_next_by_name = {}
        self.sid_sequence_to_items_by_name = {}
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
        self.hash_slot_sizes = None
        self.hash_slot_offsets = None
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
        compiled_embedding_path = self.compile_dir / 'embeddings.npy'
        if self.config.compile_config.embedding:
            if not compiled_embedding_path.exists():
                raise FileNotFoundError(f'Compiled fused embedding matrix not found: {compiled_embedding_path}')
            matrix = np.load(compiled_embedding_path).astype(np.float32)
            return torch.tensor(matrix, dtype=torch.float32)
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
        ]
        if self.config.compile_config.representation_graph:
            required_paths.extend(
                self.compile_dir / 'item_views' / f'{name}.parquet'
                for name in self.config.compile_config.representation_names
            )
            required_paths.extend(
                self.compile_dir / 'embeddings' / f'{name}.npy'
                for name in self.config.compile_config.names_for_kind('embedding')
            )
            required_paths.extend(
                self.compile_dir / 'vocab' / f'{name}.json'
                for name in self.config.compile_config.names_for_kind('sid')
            )
        else:
            required_paths.append(self.compile_dir / 'item_views' / 'uid.parquet')
        if not self.config.compile_config.representation_graph and 'embedding' in self.config.compile_config.used_views and self.config.compile_config.embedding:
            required_paths.append(self.compile_dir / 'embeddings.npy')
        self._ensure_required_paths(required_paths)

        self.meta = self._read_json(self.compile_dir / 'meta.json')
        self.vocab_meta = self._read_json(self.compile_dir / 'vocab' / 'meta.json')
        self.special_vocab = self._read_json(self.compile_dir / 'vocab' / 'special.json')
        self.prompt_main = self._read_json(self.compile_dir / 'prompts' / 'main.json')
        self.uid_raw_items = self._read_json(self.compile_dir / 'vocab' / 'uid.json')['raw_item_ids']
        self.finetune = pd.read_parquet(self.compile_dir / 'samples' / 'finetune.parquet')
        self.valid = pd.read_parquet(self.compile_dir / 'samples' / 'valid.parquet')
        self.test = pd.read_parquet(self.compile_dir / 'samples' / 'test.parquet')

        item_view_meta = self._read_json(self.compile_dir / 'item_views' / 'meta.json')
        view_names = item_view_meta.get('views') or ['uid', 'text', 'sid', 'hash', 'embedding']
        self.representation_types = item_view_meta.get('types') or {name: name for name in view_names}
        for view_name in view_names:
            values = self._load_view(view_name)
            if values is not None:
                self.item_views[view_name] = values

        if self.config.compile_config.representation_graph:
            for kind in ('uid', 'sid', 'hash', 'text', 'embedding'):
                name = self.config.compile_config.primary_name(kind)
                if name and name in self.item_views:
                    self.item_views.setdefault(kind, self.item_views[name])

        sid_entries = [entry for entry in self.vocab_meta['namespaces'] if entry['kind'] == 'sid']
        for entry in sid_entries:
            name = entry['name']
            sid_vocab_path = self.compile_dir / 'vocab' / f'{name}.json'
            if not sid_vocab_path.exists() and name == 'sid':
                sid_vocab_path = self.compile_dir / 'vocab' / 'sid.json'
            if not sid_vocab_path.exists():
                raise FileNotFoundError(f'SID vocabulary not found for {name}: {sid_vocab_path}')
            sid_vocab = self._read_json(sid_vocab_path)
            self.sid_metadata[name] = {
                **sid_vocab,
                'vocab_size': int(entry['size']),
                'num_quantizers': int(sid_vocab['num_quantizers']),
                'base_num_quantizers': int(
                    sid_vocab.get('base_num_quantizers', sid_vocab['num_quantizers'])
                ),
                'codebook_size': int(sid_vocab['codebook_size']),
                'collision_vocab_size': int(sid_vocab.get('collision_vocab_size', 0)),
                'collision_token_offset': int(sid_vocab.get('collision_token_offset', 0)),
            }
            self._build_sid_indices(name)
        if self.sid_metadata:
            primary_sid = (
                self.config.compile_config.primary_name('sid', targets=True)
                or self.config.compile_config.primary_name('sid')
                or next(iter(self.sid_metadata))
            )
            self._set_primary_sid(primary_sid)

        hash_vocab_path = self.compile_dir / 'vocab' / 'hash.json'
        if hash_vocab_path.exists():
            hash_vocab = self._read_json(hash_vocab_path)
            self.hash_num_tokens = int(hash_vocab['num_tokens'])
            self.hash_base_num_tokens = int(hash_vocab.get('base_num_tokens', self.hash_num_tokens))
            self.hash_codebook_size = int(hash_vocab['codebook_size'])
            self.hash_slot_sizes = [int(size) for size in hash_vocab.get('slot_sizes', [])]
            self.hash_slot_offsets = [int(offset) for offset in hash_vocab.get('slot_offsets', [])]
            self.hash_collision_vocab_size = int(hash_vocab.get('collision_vocab_size', 0))
            self.hash_collision_token_offset = int(hash_vocab.get('collision_token_offset', 0))
            self.hash_quantizer_name = hash_vocab.get('quantizer_name')
            self.hash_quantizer_scheme = hash_vocab.get('quantizer_scheme')
            self.hash_recommended_decoding = hash_vocab.get('recommended_decoding')
            self._build_hash_indices()

        if self.config.compile_config.representation_graph:
            for name in self.config.compile_config.names_for_kind('embedding'):
                matrix = np.load(self.compile_dir / 'embeddings' / f'{name}.npy').astype(np.float32)
                self.embedding_matrices[name] = torch.tensor(matrix, dtype=torch.float32)
            target_embedding = self.config.compile_config.primary_name('embedding', targets=True)
            fallback_embedding = self.config.compile_config.primary_name('embedding')
            primary_embedding = target_embedding or fallback_embedding
            self.embedding_matrix = self.embedding_matrices.get(primary_embedding)
        else:
            self.embedding_matrix = self._load_embedding_matrix()
        return self

    def _build_sid_indices(self, name='sid'):
        sid_view = self.item_views.get(name)
        if sid_view is None:
            self.sid_prefix_to_next_by_name[name] = {}
            self.sid_sequence_to_items_by_name[name] = {}
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

        self.sid_prefix_to_next_by_name[name] = {
            prefix: sorted(next_codes)
            for prefix, next_codes in prefix_to_next.items()
        }
        self.sid_sequence_to_items_by_name[name] = sequence_to_items

    def _set_primary_sid(self, name):
        meta = self.sid_metadata[name]
        self.sid_num_quantizers = meta['num_quantizers']
        self.sid_base_num_quantizers = meta['base_num_quantizers']
        self.sid_codebook_size = meta['codebook_size']
        self.sid_collision_vocab_size = meta['collision_vocab_size']
        self.sid_collision_token_offset = meta['collision_token_offset']
        self.sid_quantizer_name = meta.get('quantizer_name')
        self.sid_quantizer_scheme = meta.get('quantizer_scheme')
        self.sid_recommended_decoding = meta.get('recommended_decoding')
        self.sid_prefix_to_next = self.sid_prefix_to_next_by_name.get(name, {})
        self.sid_sequence_to_items = self.sid_sequence_to_items_by_name.get(name, {})

    def sid_metadata_for(self, name=None):
        if name is None:
            name = (
                self.config.compile_config.primary_name('sid', targets=True)
                or self.config.compile_config.primary_name('sid')
                or next(iter(self.sid_metadata), None)
            )
        if name not in self.sid_metadata:
            raise KeyError(f'unknown SID representation: {name}')
        return self.sid_metadata[name]

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
        if not self.sid_metadata:
            return 0
        return int(self.sid_metadata_for()['vocab_size'])

    def sid_vocab_size_for(self, name):
        return int(self.sid_metadata_for(name)['vocab_size'])

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
