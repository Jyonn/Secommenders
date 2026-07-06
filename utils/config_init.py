import os

import refconfig
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
            'task': 'task_type',
            'representation.target': 'task_type',
            'representation.source_model': 'repr_source_model',
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
            'hash.quantizer.config.num_bits': 'hash_num_bits',
            'hash.quantizer.config.num_tables': 'hash_num_tables',
            'uid.clusterer.levels': 'uid_cluster_levels',
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

        config = RefConfig().add(refconfig.CType.SMART, **kwargs)
        config = config.add(refconfig.CType.RAW).parse()

        config = Obj(config)

        for makedir in self.makedirs:
            dir_name = config[makedir]
            os.makedirs(dir_name, exist_ok=True)

        return config

    def parse(self):
        kwargs = argparse()
        return self.parse_kwargs(kwargs)
