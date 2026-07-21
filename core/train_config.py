from dataclasses import dataclass, asdict
from typing import Optional

from utils import function
from utils import model as model_utils
from utils.compile import CompileConfig, canonicalize_repr_type, compact_float, normalize_model_name, short_config_hash
from utils.experiment_template import (
    CLUSTERER_CONFIG_DEFAULTS,
    CLUSTERER_EMBEDDING_DEFAULTS,
    CLUSTERER_WORD2VEC_DEFAULTS,
    HASH_QUANTIZER_CONFIG_DEFAULTS,
    QUANTIZER_TRAINER_DEFAULTS,
    SID_ENCODER_CONFIG_DEFAULTS,
    SID_QUANTIZER_CONFIG_DEFAULTS,
    build_default_upstreams,
    merge_defaults,
    normalize_list,
    normalize_metrics,
    normalize_optional_string,
    used_upstreams_for_config,
)


def _get(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _plain_section(section, defaults: dict):
    payload = {}
    for key in defaults:
        value = _get(section, key, None) if section is not None else None
        if value is not None:
            payload[key] = value
    return payload


@dataclass
class TrainConfig:
    data: str
    model: str
    repr_type: str
    repr_source_model: Optional[str]
    sid_export: Optional[str]
    sid_coder: Optional[str]
    hash_coder: Optional[str]
    repr_combine: str
    task_type: str
    maxitems: int
    model_max_length: Optional[int]
    item_text_max_tokens: int
    batch_size: int
    accumulate_batch: int
    valid_only: int
    test_only: bool
    load_ckpt: Optional[str]
    overwrite: str
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: Optional[str]
    num_gpus: int
    freeze_backbone: str
    uid_decoding: str
    uid_cluster_levels: Optional[str]
    uid_cluster_topk: Optional[str]
    code_decoding: str
    main_metric: str
    metrics: list[str]
    patience: int
    alignment_weight: float
    code_beam_width: int
    code_beam_chunk_size: int
    code_collision_loss_weight: float
    model_dtype: str
    use_lora: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_layers: Optional[str]
    lora_target_modules: str
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    upstreams: dict

    @property
    def effective_batch_size(self):
        return int(self.batch_size) * int(self.accumulate_batch)

    @classmethod
    def from_refconfig(cls, configurations):
        root = configurations.config
        data_config = root.data
        representation = _get(root, 'representation')
        upstreams_section = _get(root, 'upstreams')
        decoder = _get(root, 'decoder')
        trainer = configurations.config.trainer
        evaluator = configurations.config.evaluator
        model = configurations.config.model
        lora = model.lora
        scratch = _get(model, 'scratch') or model.config
        raw_repr_type = _get(representation, 'history', _get(data_config, 'repr_type', None))
        raw_task_type = str(_get(representation, 'target', _get(data_config, 'task_type'))).lower()
        normalized_repr_type = canonicalize_repr_type(raw_task_type, raw_repr_type)
        raw_metrics = getattr(evaluator, 'metrics', [])
        metrics = normalize_metrics(raw_metrics)
        decoder_uid = _get(decoder, 'uid')
        decoder_sid = _get(decoder, 'sid')
        uid_decoding = str(_get(decoder_uid, 'mode', getattr(trainer, 'uid_decoding', 'flat'))).strip().lower()
        uid_cluster_topk = normalize_optional_string(_get(decoder_uid, 'topk', getattr(trainer, 'uid_cluster_topk', None)))
        repr_source_model = normalize_model_name(_get(representation, 'source_model', _get(data_config, 'repr_source_model', None)))
        sid_section = _get(upstreams_section, 'sid')
        sid_quantizer = _get(sid_section, 'quantizer')
        sid_encoder = _get(sid_section, 'encoder')
        sid_upstream_trainer = _get(sid_section, 'trainer')
        hash_section = _get(upstreams_section, 'hash')
        hash_quantizer = _get(hash_section, 'quantizer')
        uid_section = _get(upstreams_section, 'uid')
        uid_clusterer = _get(uid_section, 'clusterer')
        uid_cluster_levels = normalize_optional_string(
            _get(uid_clusterer, 'levels', getattr(trainer, 'uid_cluster_levels', None))
        )
        sid_quantizer_config = merge_defaults(
            SID_QUANTIZER_CONFIG_DEFAULTS,
            _plain_section(_get(sid_quantizer, 'config'), SID_QUANTIZER_CONFIG_DEFAULTS),
        )
        sid_encoder_config = merge_defaults(
            SID_ENCODER_CONFIG_DEFAULTS,
            _plain_section(_get(sid_encoder, 'config'), SID_ENCODER_CONFIG_DEFAULTS),
        )
        sid_quantizer_trainer = merge_defaults(
            QUANTIZER_TRAINER_DEFAULTS,
            _plain_section(sid_upstream_trainer, QUANTIZER_TRAINER_DEFAULTS),
        )
        sid_quantizer_trainer['save_best_by'] = normalize_list(sid_quantizer_trainer.get('save_best_by')) or [
            'loss',
            'coll',
            'codes',
            'recon',
        ]
        hash_quantizer_config = merge_defaults(
            HASH_QUANTIZER_CONFIG_DEFAULTS,
            _plain_section(_get(hash_quantizer, 'config'), HASH_QUANTIZER_CONFIG_DEFAULTS),
        )
        uid_cluster_word2vec = merge_defaults(
            CLUSTERER_WORD2VEC_DEFAULTS,
            _plain_section(_get(uid_clusterer, 'word2vec'), CLUSTERER_WORD2VEC_DEFAULTS),
        )
        uid_cluster_config = merge_defaults(
            CLUSTERER_CONFIG_DEFAULTS,
            _plain_section(_get(uid_clusterer, 'cluster'), CLUSTERER_CONFIG_DEFAULTS),
        )
        uid_cluster_embedding = merge_defaults(
            CLUSTERER_EMBEDDING_DEFAULTS,
            _plain_section(_get(uid_clusterer, 'embedding'), CLUSTERER_EMBEDDING_DEFAULTS),
        )
        sid_coder = str(
            _get(sid_quantizer, 'name', getattr(data_config, 'sid_coder', '')) or ''
        ).strip().lower() or None
        hash_coder = str(
            _get(hash_quantizer, 'name', getattr(data_config, 'hash_coder', '')) or ''
        ).strip().lower() or None
        sid_export = str(_get(sid_section, 'export', getattr(data_config, 'sid_export', None)) or '').strip().lower() or None
        flat_for_upstreams = {
            'repr_source_model': repr_source_model,
            'sid_coder': sid_coder,
            'sid_export': sid_export,
            'sid_embedding_model': _get(sid_section, 'embedding_model', None),
            'sid_quantizer_config': sid_quantizer_config,
            'sid_encoder_name': _get(sid_encoder, 'name', 'mlp'),
            'sid_encoder_config': sid_encoder_config,
            'sid_quantizer_trainer': sid_quantizer_trainer,
            'hash_coder': hash_coder,
            'hash_embedding_model': _get(hash_section, 'embedding_model', None),
            'hash_quantizer_config': hash_quantizer_config,
            'uid_cluster_levels': uid_cluster_levels,
            'uid_cluster_embedding': uid_cluster_embedding,
            'uid_cluster_word2vec': uid_cluster_word2vec,
            'uid_cluster_config': uid_cluster_config,
        }
        canonical_upstreams = build_default_upstreams(flat_for_upstreams)
        raw_valid_only = getattr(trainer, 'valid_only', False)
        if isinstance(raw_valid_only, bool):
            valid_only = -1 if raw_valid_only else 0
        elif raw_valid_only is None:
            valid_only = 0
        else:
            valid_only = int(raw_valid_only)
            if valid_only < 0:
                raise ValueError('trainer.valid_only must be false/0, true, or a positive integer')
        test_only = bool(getattr(trainer, 'test_only', False))
        load_ckpt = function.normalize_optional_string(getattr(trainer, 'load_ckpt', None))
        overwrite = str(getattr(trainer, 'overwrite', 'auto') or 'auto').strip().lower()
        if overwrite not in {'auto', 'true', 'false'}:
            raise ValueError('trainer.overwrite must be one of: auto, true, false')
        if valid_only and test_only:
            raise ValueError('trainer.valid_only and trainer.test_only cannot both be true')
        if load_ckpt and not test_only:
            raise ValueError('trainer.load_ckpt is only supported together with trainer.test_only=true')
        if uid_decoding == 'hierarchical' and raw_task_type != 'uid':
            raise ValueError('trainer.uid_decoding=hierarchical is only supported when task_type=uid')
        if uid_decoding == 'hierarchical' and not uid_cluster_levels:
            raise ValueError('trainer.uid_cluster_levels is required when uid_decoding=hierarchical')
        if uid_decoding == 'hierarchical' and not uid_cluster_topk:
            raise ValueError('trainer.uid_cluster_topk is required when uid_decoding=hierarchical')
        code_beam_width = int(_get(decoder_sid, 'beam_width', getattr(trainer, 'code_beam_width', 20)))
        code_beam_chunk_size = int(
            _get(decoder_sid, 'beam_chunk_size', getattr(trainer, 'code_beam_chunk_size', 0))
        )
        if code_beam_chunk_size <= 0:
            code_beam_chunk_size = max(int(trainer.batch_size), code_beam_width * 4)
        return cls(
            data=data_config.name.lower(),
            model=model.name.lower(),
            repr_type=normalized_repr_type,
            repr_source_model=repr_source_model,
            sid_export=sid_export,
            sid_coder=sid_coder,
            hash_coder=hash_coder,
            repr_combine=str(_get(representation, 'combine', getattr(data_config, 'repr_combine', 'concat'))).lower(),
            task_type=raw_task_type,
            maxitems=int(_get(representation, 'max_items', getattr(data_config, 'maxitems', 50))),
            model_max_length=int(_get(representation, 'model_max_length', model.max_length)) or None,
            item_text_max_tokens=int(_get(representation, 'item_text_max_tokens', getattr(data_config, 'item_text_max_tokens', 20))),
            batch_size=int(trainer.batch_size),
            accumulate_batch=max(1, int(getattr(trainer, 'accumulate_batch', 1))),
            valid_only=valid_only,
            test_only=test_only,
            load_ckpt=load_ckpt,
            overwrite=overwrite,
            epochs=int(trainer.epochs),
            learning_rate=float(trainer.learning_rate),
            weight_decay=float(trainer.weight_decay),
            seed=int(trainer.seed),
            device=trainer.device,
            num_gpus=int(getattr(trainer, 'num_gpus', 1)),
            freeze_backbone=str(_get(model, 'freeze_backbone', getattr(trainer, 'freeze_backbone', 'auto'))).lower(),
            uid_decoding=uid_decoding,
            uid_cluster_levels=uid_cluster_levels,
            uid_cluster_topk=uid_cluster_topk,
            code_decoding=str(_get(decoder_sid, 'mode', getattr(trainer, 'code_decoding', 'auto'))).strip().lower(),
            main_metric=str(getattr(evaluator, 'main_metric', 'ndcg@10')).strip().lower(),
            metrics=metrics,
            patience=int(evaluator.patience),
            alignment_weight=float(getattr(trainer, 'alignment', 0)),
            code_beam_width=code_beam_width,
            code_beam_chunk_size=code_beam_chunk_size,
            code_collision_loss_weight=float(
                _get(decoder_sid, 'collision_loss_weight', getattr(trainer, 'code_collision_loss_weight', 0.1))
            ),
            model_dtype=str(model.dtype).lower(),
            use_lora=str(lora.use).lower(),
            lora_rank=int(lora.rank),
            lora_alpha=int(lora.alpha),
            lora_dropout=float(lora.dropout),
            lora_layers=function.normalize_lora_layers(getattr(lora, 'layers', None)),
            lora_target_modules=str(getattr(lora, 'target_modules', 'all-linear')).strip().lower(),
            hidden_size=int(scratch.hidden_size),
            num_layers=int(scratch.num_layers),
            num_heads=int(scratch.num_heads),
            dropout=float(scratch.dropout),
            upstreams=canonical_upstreams,
        )

    @property
    def compile_config(self):
        return CompileConfig(
            data=self.data,
            model=self.model,
            repr_type=self.repr_type,
            repr_source_model=self.repr_source_model,
            sid_export=self.sid_export,
            sid_coder=self.sid_coder,
            hash_coder=self.hash_coder,
            task_type=self.task_type,
            maxitems=self.maxitems,
            model_max_length=self.model_max_length,
            item_text_max_tokens=self.item_text_max_tokens,
            repr_combine=self.repr_combine,
            upstreams=self.upstreams,
        )

    @property
    def run_id(self):
        is_llm = model_utils.match(self.model) is not None
        parts = [
            self.model,
            f'{self.repr_type}2{self.task_type}',
            f'ebs{self.effective_batch_size}',
            (
                'validonly'
                if self.valid_only == -1
                else f'validonly{self.valid_only}' if self.valid_only > 0 else None
            ),
            'testonly' if self.test_only else None,
            f'lr{compact_float(self.learning_rate)}',
            f'wd{compact_float(self.weight_decay)}',
        ]
        parts = [part for part in parts if part]
        if self.repr_combine != 'concat':
            parts.append(self.repr_combine)
        if self.task_type == 'uid' and self.uid_decoding != 'flat':
            parts.append(f'ud-{self.uid_decoding}')
        if self.task_type == 'uid' and self.uid_decoding == 'hierarchical' and self.uid_cluster_levels:
            parts.append(f'ucl-{self.uid_cluster_levels.replace(",", "x")}')
        if self.task_type == 'uid' and self.uid_decoding == 'hierarchical' and self.uid_cluster_topk:
            parts.append(f'uck-{self.uid_cluster_topk.replace(",", "x")}')
        if self.freeze_backbone != 'auto':
            parts.append(f'fr-{self.freeze_backbone}')
        if self.use_lora != 'auto':
            parts.append(f'lo-{self.use_lora}')
        if self.alignment_weight > 0 and self.repr_combine != 'add':
            parts.append(f'al{compact_float(self.alignment_weight)}')
        if self.task_type == 'sid' and self.code_decoding != 'auto':
            parts.append(f'cd-{self.code_decoding}')
        if self.task_type == 'sid' and self.code_beam_width != 20:
            parts.append(f'cb{self.code_beam_width}')
        if self.task_type in {'sid', 'hash'} and self.code_collision_loss_weight != 0.1:
            parts.append(f'ccw{compact_float(self.code_collision_loss_weight)}')
        if is_llm:
            parts.extend(
                [
                    f'r{self.lora_rank}',
                    f'a{self.lora_alpha}',
                    f'd{compact_float(self.lora_dropout)}',
                ]
            )
            if self.lora_layers:
                parts.append(f'ly{self.lora_layers}')
        parts.append(f'h{short_config_hash(self.sign_payload)}')
        return '__'.join(parts)

    @property
    def sign_payload(self):
        payload = asdict(self)
        if self.model == 'scratch':
            payload['backbone_architecture'] = 'llama-v1'
        elif self.model == 'scratchlegacy':
            payload['backbone_architecture'] = 'torch-transformer-v1'
        payload.pop('device', None)
        payload['effective_batch_size'] = self.effective_batch_size
        payload.pop('seed', None)
        payload.pop('batch_size', None)
        payload.pop('accumulate_batch', None)
        payload.pop('valid_only', None)
        payload.pop('test_only', None)
        payload.pop('load_ckpt', None)
        payload.pop('overwrite', None)
        payload.pop('code_beam_chunk_size', None)
        used_views = self.compile_config.used_views
        used_upstreams = used_upstreams_for_config(self.task_type, self.repr_type, self.uid_decoding)
        payload['upstreams'] = {
            key: value for key, value in self.upstreams.items()
            if key in used_upstreams
        }
        if not any(view in {'sid', 'hash', 'embedding'} for view in used_views):
            payload.pop('repr_source_model', None)
        if 'sid' not in used_views:
            payload.pop('sid_export', None)
            payload.pop('sid_coder', None)
        if 'hash' not in used_views:
            payload.pop('hash_coder', None)
        if self.task_type != 'uid':
            payload.pop('uid_decoding', None)
            payload.pop('uid_cluster_levels', None)
            payload.pop('uid_cluster_topk', None)
        elif self.uid_decoding != 'hierarchical':
            payload.pop('uid_cluster_levels', None)
            payload.pop('uid_cluster_topk', None)
        if self.task_type != 'sid':
            payload.pop('code_decoding', None)
            payload.pop('code_beam_width', None)
            payload.pop('code_beam_chunk_size', None)
        if not any(view in {'sid', 'hash'} for view in used_views):
            payload.pop('code_collision_loss_weight', None)
        if self.repr_combine == 'add' or len(self.compile_config.repr_types) <= 1:
            payload.pop('alignment_weight', None)
        if not payload.get('upstreams'):
            payload.pop('upstreams', None)
        return payload
