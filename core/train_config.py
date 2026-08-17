from dataclasses import dataclass, asdict, replace
from types import SimpleNamespace
from typing import Optional

from oba import NotFound

from utils import function
from utils import model as model_utils
from utils.compile import (
    CompileConfig,
    canonicalize_repr_type,
    canonicalize_task_type,
    compact_float,
    normalize_model_name,
    short_config_hash,
)
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
from utils.frequency_breakdown import normalize_frequency_boundaries
from utils.embedding_fusion import embedding_fusion_from_flat, normalize_embedding_fusion
from utils.word2vec import WORD2VEC_DEFAULTS
from utils.representation_schema import (
    graph_from_legacy_config,
    normalize_representation_graph,
    plain,
    upstreams_from_graph,
)


def _get(obj, name, default=None):
    try:
        value = getattr(obj, name)
    except Exception:
        return default
    return default if isinstance(value, NotFound) else value


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
    repr_embedding: Optional[dict]
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
    frequency_breakdown: bool
    frequency_buckets: list[int]
    overwrite: str
    epochs: int
    learning_rate: float
    weight_decay: float
    lr_scheduler: str
    warmup_ratio: float
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
    multi_candidate_topk: int
    multi_output_topk: int
    multi_fusion: str
    multi_uid_weight: float
    multi_score_normalization: str
    multi_temperature_uid: float
    multi_temperature_sid: float
    multi_frequency_threshold: float
    multi_frequency_smoothing: float
    multi_uid_loss_weight: float
    multi_sid_loss_weight: float
    multi_fused_loss_weight: float
    multi_consistency_weight: float
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
    representation_graph: Optional[dict] = None

    @property
    def effective_batch_size(self):
        return int(self.batch_size) * int(self.accumulate_batch)

    @property
    def task_types(self):
        return self.task_type.split('+')

    @property
    def is_multi_task(self):
        return len(self.task_types) > 1

    @classmethod
    def from_refconfig(cls, configurations):
        root = configurations.config
        if str(_get(root, 'schema_version', '') or '').strip().lower() == 'trainer.v4':
            return cls._from_v4(root)
        data_config = root.data
        representation = _get(root, 'representation')
        upstreams_section = _get(root, 'upstreams')
        decoder = _get(root, 'decoder')
        trainer = configurations.config.trainer
        evaluator = configurations.config.evaluator
        model = configurations.config.model
        lora = model.lora
        # model.config was the former section name. Keep it as a read fallback
        # for external legacy YAML files; canonical configs use model.scratch.
        scratch = _get(model, 'scratch') or _get(model, 'config')
        if scratch is None:
            raise ValueError('model.scratch configuration is required')
        raw_repr_type = _get(representation, 'history', _get(data_config, 'repr_type', None))
        raw_task_type = canonicalize_task_type(
            _get(representation, 'target')
            or _get(representation, 'legacy_target')
            or _get(data_config, 'task_type')
        )
        normalized_repr_type = canonicalize_repr_type(raw_task_type, raw_repr_type)
        raw_metrics = getattr(evaluator, 'metrics', [])
        metrics = normalize_metrics(raw_metrics)
        decoder_uid = _get(decoder, 'uid')
        decoder_sid = _get(decoder, 'sid')
        decoder_multi = _get(decoder, 'multi')
        multi_fusion = _get(decoder_multi, 'fusion')
        multi_frequency = _get(decoder_multi, 'frequency')
        trainer_multi = _get(trainer, 'multi')
        uid_decoding = str(_get(decoder_uid, 'mode', getattr(trainer, 'uid_decoding', 'flat'))).strip().lower()
        uid_cluster_topk = normalize_optional_string(_get(decoder_uid, 'topk', getattr(trainer, 'uid_cluster_topk', None)))
        repr_source_model = normalize_model_name(_get(representation, 'source_model', _get(data_config, 'repr_source_model', None)))
        repr_embedding_section = _get(representation, 'embedding')
        sid_section = _get(upstreams_section, 'sid')
        sid_quantizer = _get(sid_section, 'quantizer')
        sid_encoder = _get(sid_section, 'encoder')
        sid_upstream_trainer = _get(sid_section, 'trainer')
        sid_embedding = _get(sid_section, 'embedding')
        hash_section = _get(upstreams_section, 'hash')
        hash_quantizer = _get(hash_section, 'quantizer')
        hash_embedding = _get(hash_section, 'embedding')
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
        sid_embedding_config = embedding_fusion_from_flat(
            _get(sid_embedding, 'models'),
            normalize=_get(sid_embedding, 'normalize'),
            reduce_dims=_get(sid_embedding, 'reduce_dims'),
            weights=_get(sid_embedding, 'weights'),
            fusion=_get(sid_embedding, 'fusion', 'concat'),
            normalize_output=_get(sid_embedding, 'normalize_output', False),
            word2vec_config=_plain_section(_get(sid_embedding, 'word2vec'), WORD2VEC_DEFAULTS),
        )
        repr_embedding_config = embedding_fusion_from_flat(
            _get(repr_embedding_section, 'models'),
            normalize=_get(repr_embedding_section, 'normalize'),
            reduce_dims=_get(repr_embedding_section, 'reduce_dims'),
            weights=_get(repr_embedding_section, 'weights'),
            fusion=_get(repr_embedding_section, 'fusion', 'concat'),
            normalize_output=_get(repr_embedding_section, 'normalize_output', False),
            word2vec_config=_plain_section(_get(repr_embedding_section, 'word2vec'), WORD2VEC_DEFAULTS),
        )
        if repr_embedding_config:
            repr_embedding_config = normalize_embedding_fusion(repr_embedding_config)
        hash_embedding_config = embedding_fusion_from_flat(
            _get(hash_embedding, 'models'),
            normalize=_get(hash_embedding, 'normalize'),
            reduce_dims=_get(hash_embedding, 'reduce_dims'),
            weights=_get(hash_embedding, 'weights'),
            fusion=_get(hash_embedding, 'fusion', 'concat'),
            normalize_output=_get(hash_embedding, 'normalize_output', False),
        )
        flat_for_upstreams = {
            'repr_source_model': repr_source_model,
            'sid_coder': sid_coder,
            'sid_export': sid_export,
            'sid_embedding_model': _get(sid_section, 'embedding_model', None),
            'sid_embedding': sid_embedding_config,
            'sid_quantizer_config': sid_quantizer_config,
            'sid_encoder_name': _get(sid_encoder, 'name', 'mlp'),
            'sid_encoder_config': sid_encoder_config,
            'sid_quantizer_trainer': sid_quantizer_trainer,
            'hash_coder': hash_coder,
            'hash_embedding_model': _get(hash_section, 'embedding_model', None),
            'hash_embedding': hash_embedding_config,
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
        raw_frequency_breakdown = getattr(evaluator, 'frequency_breakdown', False)
        frequency_breakdown = (
            raw_frequency_breakdown
            if isinstance(raw_frequency_breakdown, bool)
            else function.coerce_bool(str(raw_frequency_breakdown).strip().lower(), default=False)
        )
        frequency_buckets = normalize_frequency_boundaries(
            getattr(evaluator, 'frequency_buckets', None)
        )
        overwrite = str(getattr(trainer, 'overwrite', 'auto') or 'auto').strip().lower()
        if overwrite not in {'auto', 'true', 'false'}:
            raise ValueError('trainer.overwrite must be one of: auto, true, false')
        if valid_only and test_only:
            raise ValueError('trainer.valid_only and trainer.test_only cannot both be true')
        if load_ckpt and not test_only:
            raise ValueError('trainer.load_ckpt is only supported together with trainer.test_only=true')
        if frequency_breakdown and not test_only:
            raise ValueError('evaluator.frequency_breakdown requires trainer.test_only=true')
        if frequency_breakdown and not load_ckpt:
            raise ValueError('evaluator.frequency_breakdown requires trainer.load_ckpt')
        lr_scheduler = str(getattr(trainer, 'lr_scheduler', 'constant')).strip().lower()
        if lr_scheduler not in {'constant', 'linear', 'cosine'}:
            raise ValueError('trainer.lr_scheduler must be one of: constant, linear, cosine')
        warmup_ratio = float(getattr(trainer, 'warmup_ratio', 0.0))
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError('trainer.warmup_ratio must be in [0, 1)')
        if int(trainer.epochs) <= 0 and (warmup_ratio > 0 or lr_scheduler != 'constant'):
            raise ValueError(
                'trainer.epochs must be positive when warmup_ratio > 0 or '
                'lr_scheduler is linear/cosine'
            )
        task_types = raw_task_type.split('+')
        supported_task_types = {'uid', 'sid', 'hash', 'embedding'}
        unsupported_task_types = [task for task in task_types if task not in supported_task_types]
        if unsupported_task_types:
            raise ValueError(f'unsupported representation.target entries: {unsupported_task_types}')
        if len(task_types) > 1 and task_types != ['sid', 'uid']:
            raise ValueError('multi target decoding currently supports exactly representation.target=sid+uid')
        if uid_decoding == 'hierarchical' and task_types != ['uid']:
            raise ValueError('trainer.uid_decoding=hierarchical is only supported for a uid-only target')
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
        multi_candidate_topk = int(_get(decoder_multi, 'candidate_topk', 100))
        multi_output_topk = int(_get(decoder_multi, 'output_topk', 20))
        if multi_candidate_topk <= 0 or multi_output_topk <= 0:
            raise ValueError('decoder.multi candidate_topk and output_topk must be positive')
        multi_fusion_mode = str(_get(multi_fusion, 'mode', 'frequency')).strip().lower()
        if multi_fusion_mode not in {'fixed', 'frequency'}:
            raise ValueError('decoder.multi.fusion.mode must be fixed or frequency')
        multi_score_normalization = str(_get(multi_fusion, 'score_normalization', 'zscore')).strip().lower()
        if multi_score_normalization not in {'none', 'zscore', 'minmax'}:
            raise ValueError('decoder.multi.fusion.score_normalization must be none, zscore, or minmax')
        multi_uid_weight = float(_get(multi_fusion, 'uid_weight', 0.5))
        if not 0.0 <= multi_uid_weight <= 1.0:
            raise ValueError('decoder.multi.fusion.uid_weight must be in [0, 1]')
        multi_temperature_uid = float(_get(multi_fusion, 'temperature_uid', 1.0))
        multi_temperature_sid = float(_get(multi_fusion, 'temperature_sid', 1.0))
        if multi_temperature_uid <= 0 or multi_temperature_sid <= 0:
            raise ValueError('decoder.multi fusion temperatures must be positive')
        multi_frequency_smoothing = float(_get(multi_frequency, 'smoothing', 2.0))
        if multi_frequency_smoothing <= 0:
            raise ValueError('decoder.multi.frequency.smoothing must be positive')
        multi_loss_values = {
            'uid': float(_get(trainer_multi, 'uid_loss_weight', 1.0)),
            'sid': float(_get(trainer_multi, 'sid_loss_weight', 1.0)),
            'fused': float(_get(trainer_multi, 'fused_loss_weight', 0.0)),
            'consistency': float(_get(trainer_multi, 'consistency_weight', 0.0)),
        }
        if any(value < 0 for value in multi_loss_values.values()):
            raise ValueError('trainer.multi loss weights must be non-negative')
        if len(task_types) > 1 and multi_loss_values['fused'] > 0:
            raise ValueError('trainer.multi.fused_loss_weight is reserved for a future differentiable fusion loss')
        if len(task_types) > 1 and multi_loss_values['consistency'] > 0:
            raise ValueError('trainer.multi.consistency_weight is reserved for a future cross-head consistency loss')
        config = cls(
            data=data_config.name.lower(),
            model=model.name.lower(),
            repr_type=normalized_repr_type,
            repr_source_model=repr_source_model,
            repr_embedding=repr_embedding_config,
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
            frequency_breakdown=frequency_breakdown,
            frequency_buckets=frequency_buckets,
            overwrite=overwrite,
            epochs=int(trainer.epochs),
            learning_rate=float(trainer.learning_rate),
            weight_decay=float(trainer.weight_decay),
            lr_scheduler=lr_scheduler,
            warmup_ratio=warmup_ratio,
            seed=int(trainer.seed),
            device=trainer.device,
            num_gpus=int(getattr(trainer, 'num_gpus', 1)),
            freeze_backbone=str(_get(model, 'freeze_backbone', getattr(trainer, 'freeze_backbone', 'auto'))).lower(),
            uid_decoding=uid_decoding,
            uid_cluster_levels=uid_cluster_levels,
            uid_cluster_topk=uid_cluster_topk,
            code_decoding=str(_get(decoder_sid, 'mode', getattr(trainer, 'code_decoding', 'auto'))).strip().lower(),
            main_metric='|'.join(
                metric.strip().lower()
                for metric in str(getattr(evaluator, 'main_metric', 'ndcg@10')).split('|')
                if metric.strip()
            ) or 'loss',
            metrics=metrics,
            patience=int(evaluator.patience),
            alignment_weight=float(getattr(trainer, 'alignment', 0)),
            code_beam_width=code_beam_width,
            code_beam_chunk_size=code_beam_chunk_size,
            code_collision_loss_weight=float(
                _get(decoder_sid, 'collision_loss_weight', getattr(trainer, 'code_collision_loss_weight', 0.1))
            ),
            multi_candidate_topk=multi_candidate_topk,
            multi_output_topk=multi_output_topk,
            multi_fusion=multi_fusion_mode,
            multi_uid_weight=multi_uid_weight,
            multi_score_normalization=multi_score_normalization,
            multi_temperature_uid=multi_temperature_uid,
            multi_temperature_sid=multi_temperature_sid,
            multi_frequency_threshold=float(_get(multi_frequency, 'threshold', 5)),
            multi_frequency_smoothing=multi_frequency_smoothing,
            multi_uid_loss_weight=multi_loss_values['uid'],
            multi_sid_loss_weight=multi_loss_values['sid'],
            multi_fused_loss_weight=multi_loss_values['fused'],
            multi_consistency_weight=multi_loss_values['consistency'],
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
        if config.repr_combine == 'add':
            return config
        graph = graph_from_legacy_config(asdict(config))
        return replace(config, representation_graph=graph, upstreams=upstreams_from_graph(graph))

    @classmethod
    def _from_v4(cls, root):
        raw = plain(root)
        graph = normalize_representation_graph(
            raw.get('representations'),
            raw.get('encoder'),
            raw.get('decoder'),
        )
        catalog = graph['representations']
        encoder_names = graph['encoder']['representations']
        targets = graph['decoder']['targets']
        target_names = [target['representation'] for target in targets]
        target_types = [catalog[name]['type'] for name in target_names]
        encoder_types = [catalog[name]['type'] for name in encoder_names]
        history_types = list(dict.fromkeys(target_types + encoder_types))
        model_name = str((raw.get('model') or {}).get('name') or '').strip().lower()
        if model_name not in {'scratch', 'scratchlegacy'} and model_utils.match(model_name) is None:
            raise ValueError(
                f'unknown model {model_name!r}; use scratch/scratchlegacy or configure the alias in .model'
            )
        if model_name in {'scratch', 'scratchlegacy'} and 'text' in encoder_types:
            raise ValueError('scratch backbones do not support text representations')

        upstreams = {}
        sid_names = [name for name in encoder_names if catalog[name]['type'] == 'sid']
        if sid_names:
            spec = catalog[sid_names[0]]
            codec = spec['codec']
            upstreams['sid'] = {
                'kind': 'quantized',
                'embedding_model': None,
                'embedding': spec['embedding'],
                'export': codec['export'],
                'quantizer': {'name': codec['name'], 'config': codec['quantizer']},
                'encoder': codec['encoder'],
                'trainer': codec['trainer'],
            }
        hash_names = [name for name in encoder_names if catalog[name]['type'] == 'hash']
        if hash_names:
            spec = catalog[hash_names[0]]
            codec = spec['codec']
            upstreams['hash'] = {
                'kind': 'quantized',
                'embedding_model': None,
                'embedding': spec['embedding'],
                'export': 'hash',
                'quantizer': {'name': codec['name'], 'config': codec['quantizer']},
            }
        uid_names = [name for name in encoder_names if catalog[name]['type'] == 'uid']
        if uid_names and catalog[uid_names[0]].get('hierarchy'):
            upstreams['uid'] = {
                'kind': 'clustered',
                'clusterer': catalog[uid_names[0]]['hierarchy'],
            }

        decoder = {
            'uid': {'mode': 'flat', 'topk': None},
            'sid': {'mode': 'auto', 'beam_width': 20, 'beam_chunk_size': 0, 'collision_loss_weight': 0.1},
            'hash': {'mode': 'parallel'},
            'multi': {
                'candidate_topk': 100,
                'output_topk': 20,
                'fusion': {
                    'mode': 'frequency', 'uid_weight': 0.5, 'score_normalization': 'zscore',
                    'temperature_uid': 1.0, 'temperature_sid': 1.0,
                },
                'frequency': {'threshold': 5, 'smoothing': 2.0},
            },
        }
        decoder['multi'].update(graph['decoder']['multiple'])
        for target in targets:
            kind = catalog[target['representation']]['type']
            decoding = dict(target.get('decoding') or {})
            if kind == 'uid':
                decoder['uid'] = decoding
            elif kind == 'sid':
                decoder['sid'] = decoding
            elif kind == 'hash':
                decoder['hash'] = decoding

        encoder = raw.get('encoder') or {}
        fallback_source_model = None
        for name in encoder_names:
            embedding_spec = catalog[name].get('embedding') or {}
            sources = embedding_spec.get('sources') or []
            if sources:
                fallback_source_model = sources[0]['model']
                break
        trainer_defaults = {
            'batch_size': 64, 'accumulate_batch': 1, 'valid_only': False, 'test_only': False,
            'load_ckpt': None, 'overwrite': 'auto', 'epochs': 0, 'learning_rate': 0.0001,
            'weight_decay': 0.01, 'lr_scheduler': 'constant', 'warmup_ratio': 0.0, 'seed': 42,
            'device': None, 'num_gpus': 1, 'alignment': 0.0, 'uid_cluster_levels': None,
            'uid_cluster_topk': None, 'code_decoding': 'auto', 'code_beam_width': 20,
            'code_beam_chunk_size': 0, 'code_collision_loss_weight': 0.1,
            'multi': {'uid_loss_weight': 1.0, 'sid_loss_weight': 1.0, 'fused_loss_weight': 0.0, 'consistency_weight': 0.0},
        }
        trainer_defaults.update(raw.get('trainer') or {})
        evaluator_defaults = {
            'main_metric': 'ndcg@10', 'patience': 3,
            'metrics': ['ndcg@5', 'ndcg@10', 'ndcg@20', 'hr@5', 'hr@10', 'hr@20', 'mrr'],
            'frequency_breakdown': False, 'frequency_buckets': [0, 5, 20, 100],
        }
        evaluator_defaults.update(raw.get('evaluator') or {})
        legacy = {
            'data': raw.get('data') or {},
            'representation': {
                'history': '+'.join(history_types),
                'target': '+'.join(target_types),
                'source_model': fallback_source_model,
                'combine': graph['encoder']['combine'],
                'max_items': encoder.get('max_items', 50),
                'model_max_length': encoder.get('model_max_length', 0),
                'item_text_max_tokens': encoder.get('item_text_max_tokens', 20),
            },
            'upstreams': upstreams,
            'model': raw.get('model') or {},
            'decoder': decoder,
            'trainer': trainer_defaults,
            'evaluator': evaluator_defaults,
        }
        def namespace(value):
            if isinstance(value, dict):
                return SimpleNamespace(**{key: namespace(item) for key, item in value.items()})
            if isinstance(value, list):
                return [namespace(item) for item in value]
            return value

        config = cls.from_refconfig(SimpleNamespace(config=namespace(legacy)))
        return replace(
            config,
            representation_graph=graph,
            upstreams=upstreams,
            repr_source_model=None,
            repr_embedding=None,
        )

    @property
    def compile_config(self):
        return CompileConfig(
            data=self.data,
            model=self.model,
            repr_type=self.repr_type,
            repr_source_model=self.repr_source_model,
            embedding=self.repr_embedding,
            sid_export=self.sid_export,
            sid_coder=self.sid_coder,
            hash_coder=self.hash_coder,
            task_type=self.task_type,
            maxitems=self.maxitems,
            model_max_length=self.model_max_length,
            item_text_max_tokens=self.item_text_max_tokens,
            repr_combine=self.repr_combine,
            upstreams=self.upstreams,
            representation_graph=self.representation_graph,
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
            f'ls-{self.lr_scheduler}' if self.lr_scheduler != 'constant' else None,
            f'wu{compact_float(self.warmup_ratio)}' if self.warmup_ratio > 0 else None,
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
        payload.pop('frequency_breakdown', None)
        payload.pop('frequency_buckets', None)
        payload.pop('overwrite', None)
        payload.pop('code_beam_chunk_size', None)
        payload.pop('multi_candidate_topk', None)
        payload.pop('multi_output_topk', None)
        if self.representation_graph:
            for key in ('repr_source_model', 'repr_embedding', 'sid_export', 'sid_coder', 'hash_coder', 'upstreams'):
                payload.pop(key, None)
        used_views = self.compile_config.used_views
        used_upstreams = used_upstreams_for_config(self.task_type, self.repr_type, self.uid_decoding)
        if not self.representation_graph:
            payload['upstreams'] = {
                key: value for key, value in self.upstreams.items()
                if key in used_upstreams
            }
        if not any(view in {'sid', 'hash', 'embedding'} for view in used_views):
            payload.pop('repr_source_model', None)
        if 'embedding' not in used_views or not payload.get('repr_embedding'):
            payload.pop('repr_embedding', None)
        if 'sid' not in used_views:
            payload.pop('sid_export', None)
            payload.pop('sid_coder', None)
        if 'hash' not in used_views:
            payload.pop('hash_coder', None)
        if 'uid' not in self.task_type.split('+'):
            payload.pop('uid_decoding', None)
            payload.pop('uid_cluster_levels', None)
            payload.pop('uid_cluster_topk', None)
        elif self.uid_decoding != 'hierarchical':
            payload.pop('uid_cluster_levels', None)
            payload.pop('uid_cluster_topk', None)
        if 'sid' not in self.task_type.split('+'):
            payload.pop('code_decoding', None)
            payload.pop('code_beam_width', None)
            payload.pop('code_beam_chunk_size', None)
        if not any(view in {'sid', 'hash'} for view in used_views):
            payload.pop('code_collision_loss_weight', None)
        if self.repr_combine == 'add' or len(self.compile_config.repr_types) <= 1:
            payload.pop('alignment_weight', None)
        if not payload.get('upstreams'):
            payload.pop('upstreams', None)
        if '+' not in self.task_type:
            for key in list(payload):
                if key.startswith('multi_'):
                    payload.pop(key, None)
        return payload
