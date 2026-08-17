import os
import tempfile
from pathlib import Path

import refconfig
import yaml
from oba import Obj
from refconfig import RefConfig

from .function import argparse


class ConfigInit:
    def __init__(self, required_args, default_args, makedirs):
        self.required_args = required_args
        self.default_args = default_args
        self.makedirs = makedirs

    @staticmethod
    def _normalize_aliases(kwargs):
        aliases = {
            'repr': 'repr_type',
            'representation.history': 'repr_type',
            'task': 'task_repr',
            'representation.target': 'task_repr',
            'representation.source_model': 'repr_source_model',
            'representation.embedding.models': 'repr_embedding_models',
            'representation.embedding.normalize': 'repr_embedding_normalize',
            'representation.embedding.reduce_dims': 'repr_embedding_reduce_dims',
            'representation.embedding.weights': 'repr_embedding_weights',
            'representation.embedding.fusion': 'repr_embedding_fusion',
            'representation.embedding.normalize_output': 'repr_embedding_normalize_output',
            'representation.combine': 'repr_combine',
            'representation.max_items': 'maxitems',
            'representation.model_max_length': 'model_max_length',
            'representation.item_text_max_tokens': 'item_text_max_tokens',
            'sid_quantizer_name': 'sid_coder',
            'sid.quantizer.name': 'sid_coder',
            'sid.quantizer.export': 'sid_export',
            'sid.export': 'sid_export',
            'sid.embedding_model': 'sid_embedding_model',
            'sid.quantizer.config.latent_dim': 'sid_latent_dim',
            'sid.quantizer.config.codebook_size': 'sid_codebook_size',
            'sid.quantizer.config.commitment_weight': 'sid_commitment_weight',
            'sid.quantizer.config.codebook_weight': 'sid_codebook_weight',
            'sid.quantizer.config.num_quantizers': 'sid_num_quantizers',
            'sid.quantizer.config.num_codebooks': 'sid_num_codebooks',
            'sid.quantizer.config.assignment_strategy': 'sid_assignment_strategy',
            'sid.quantizer.config.sinkhorn_epsilon': 'sid_sinkhorn_epsilon',
            'sid.encoder.name': 'sid_encoder_name',
            'sid.encoder.config.hidden_dims': 'sid_hidden_dims',
            'sid.trainer.epochs': 'sid_epochs',
            'sid.trainer.batch_size': 'sid_batch_size',
            'sid.trainer.learning_rate': 'sid_lr',
            'hash_quantizer_name': 'hash_coder',
            'hash.quantizer.name': 'hash_coder',
            'hash.embedding_model': 'hash_embedding_model',
            'sid.embedding.models': 'sid_embedding_models',
            'sid.embedding.normalize': 'sid_embedding_normalize',
            'sid.embedding.reduce_dims': 'sid_embedding_reduce_dims',
            'sid.embedding.weights': 'sid_embedding_weights',
            'sid.embedding.fusion': 'sid_embedding_fusion',
            'sid.embedding.normalize_output': 'sid_embedding_normalize_output',
            'sid.embedding.word2vec.vector_size': 'sid_word2vec_vector_size',
            'sid.embedding.word2vec.window': 'sid_word2vec_window',
            'sid.embedding.word2vec.patience': 'sid_word2vec_patience',
            'sid.embedding.word2vec.negative': 'sid_word2vec_negative',
            'hash.embedding.models': 'hash_embedding_models',
            'hash.embedding.normalize': 'hash_embedding_normalize',
            'hash.embedding.reduce_dims': 'hash_embedding_reduce_dims',
            'hash.embedding.weights': 'hash_embedding_weights',
            'hash.embedding.fusion': 'hash_embedding_fusion',
            'hash.embedding.normalize_output': 'hash_embedding_normalize_output',
            'hash.quantizer.config.num_bits': 'hash_num_bits',
            'hash.quantizer.config.num_tables': 'hash_num_tables',
            'uid.clusterer.levels': 'uid_cluster_levels',
            'uid.clusterer.embedding.source': 'uid_cluster_embedding_source',
            'uid.clusterer.embedding.content_model': 'uid_cluster_content_model',
            'uid.clusterer.embedding.content_reduce_dim': 'uid_cluster_content_reduce_dim',
            'uid.clusterer.embedding.normalize_blocks': 'uid_cluster_normalize_blocks',
            'uid.clusterer.embedding.mix_alpha': 'uid_cluster_mix_alpha',
            'uid.decoder.topk': 'uid_cluster_topk',
            'decoder.uid.topk': 'uid_cluster_topk',
            'decoder.uid.mode': 'uid_decoding',
            'decoder.sid.mode': 'code_decoding',
            'decoder.sid.beam_width': 'code_beam_width',
            'decoder.sid.beam_chunk_size': 'code_beam_chunk_size',
            'decoder.sid.collision_loss_weight': 'code_collision_loss_weight',
            'lr': 'learning_rate',
            'wd': 'weight_decay',
        }
        normalized = dict(kwargs)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        return normalized

    def parse_kwargs(self, kwargs):
        kwargs = self._normalize_aliases(kwargs)
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f'miss argument {arg}')

        for arg in self.default_args:
            if arg not in kwargs:
                kwargs[arg] = self.default_args[arg]

        temporary_config = None
        config_path = kwargs.get('config')
        if config_path:
            merged = self._load_extended_yaml(Path(config_path))
            if merged is not None:
                handle = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
                yaml.safe_dump(merged, handle, sort_keys=False)
                handle.close()
                temporary_config = handle.name
                kwargs['config'] = temporary_config
        try:
            config = RefConfig().add(refconfig.CType.SMART, **kwargs)
            config = config.add(refconfig.CType.RAW).parse()
        finally:
            if temporary_config:
                Path(temporary_config).unlink(missing_ok=True)

        config = Obj(config)

        for makedir in self.makedirs:
            dir_name = config[makedir]
            os.makedirs(dir_name, exist_ok=True)

        return config

    @classmethod
    def _load_extended_yaml(cls, path: Path):
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict) or not payload.get('extends'):
            return None
        parent_path = Path(str(payload.pop('extends')))
        if not parent_path.is_absolute():
            candidate = path.parent / parent_path
            parent_path = candidate if candidate.exists() else Path.cwd() / parent_path
        parent = cls._load_extended_yaml(parent_path)
        if parent is None:
            parent = yaml.safe_load(parent_path.read_text()) or {}
            parent.pop('extends', None)
        return cls._deep_merge(parent, payload)

    @classmethod
    def _deep_merge(cls, base, override):
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override
        merged = dict(base)
        for key, value in override.items():
            merged[key] = cls._deep_merge(merged[key], value) if key in merged else value
        return merged

    def parse(self):
        kwargs = argparse()
        return self.parse_kwargs(kwargs)
