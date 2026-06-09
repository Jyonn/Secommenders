from dataclasses import dataclass, asdict
from typing import Optional

from utils import function
from utils import model as model_utils
from utils.compile import CompileConfig, canonicalize_repr_type, compact_float, normalize_model_name, short_config_hash


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
    valid_only: bool
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

    @classmethod
    def from_refconfig(cls, configurations):
        data_config = configurations.config.data
        trainer = configurations.config.trainer
        evaluator = configurations.config.evaluator
        model = configurations.config.model
        lora = model.lora
        scratch = model.config
        try:
            raw_repr_type = data_config.repr_type
        except Exception:
            raw_repr_type = None
        raw_task_type = str(data_config.task_type).lower()
        normalized_repr_type = canonicalize_repr_type(raw_task_type, raw_repr_type)
        raw_metrics = getattr(evaluator, 'metrics', [])
        if isinstance(raw_metrics, str):
            metrics = [metric.strip().lower() for metric in raw_metrics.split(',') if metric.strip()]
        else:
            metrics = [str(metric).strip().lower() for metric in list(raw_metrics) if str(metric).strip()]
        uid_decoding = str(getattr(trainer, 'uid_decoding', 'flat')).strip().lower()
        uid_cluster_levels = function.normalize_optional_string(getattr(trainer, 'uid_cluster_levels', None))
        uid_cluster_topk = function.normalize_optional_string(getattr(trainer, 'uid_cluster_topk', None))
        if uid_decoding == 'hierarchical' and raw_task_type != 'uid':
            raise ValueError('trainer.uid_decoding=hierarchical is only supported when task_type=uid')
        if uid_decoding == 'hierarchical' and not uid_cluster_levels:
            raise ValueError('trainer.uid_cluster_levels is required when uid_decoding=hierarchical')
        if uid_decoding == 'hierarchical' and not uid_cluster_topk:
            raise ValueError('trainer.uid_cluster_topk is required when uid_decoding=hierarchical')
        return cls(
            data=data_config.name.lower(),
            model=model.name.lower(),
            repr_type=normalized_repr_type,
            repr_source_model=normalize_model_name(data_config.repr_source_model),
            sid_export=data_config.sid_export.lower() if data_config.sid_export else None,
            sid_coder=str(getattr(data_config, 'sid_coder', '')).strip().lower() or None,
            hash_coder=str(getattr(data_config, 'hash_coder', '')).strip().lower() or None,
            repr_combine=data_config.repr_combine.lower(),
            task_type=raw_task_type,
            maxitems=int(data_config.maxitems),
            model_max_length=int(model.max_length) or None,
            item_text_max_tokens=int(data_config.item_text_max_tokens),
            batch_size=int(trainer.batch_size),
            accumulate_batch=max(1, int(getattr(trainer, 'accumulate_batch', 1))),
            valid_only=bool(getattr(trainer, 'valid_only', False)),
            epochs=int(trainer.epochs),
            learning_rate=float(trainer.learning_rate),
            weight_decay=float(trainer.weight_decay),
            seed=int(trainer.seed),
            device=trainer.device,
            num_gpus=int(getattr(trainer, 'num_gpus', 1)),
            freeze_backbone=str(trainer.freeze_backbone).lower(),
            uid_decoding=uid_decoding,
            uid_cluster_levels=uid_cluster_levels,
            uid_cluster_topk=uid_cluster_topk,
            code_decoding=str(getattr(trainer, 'code_decoding', 'auto')).strip().lower(),
            main_metric=str(getattr(evaluator, 'main_metric', 'loss')).strip().lower(),
            metrics=metrics,
            patience=int(evaluator.patience),
            alignment_weight=float(getattr(trainer, 'alignment', 0)),
            code_beam_width=int(getattr(trainer, 'code_beam_width', 20)),
            code_beam_chunk_size=int(getattr(trainer, 'code_beam_chunk_size', 0)) or int(trainer.batch_size),
            code_collision_loss_weight=float(getattr(trainer, 'code_collision_loss_weight', 0.1)),
            model_dtype=str(model.dtype).lower(),
            use_lora=str(lora.use).lower(),
            lora_rank=int(lora.rank),
            lora_alpha=int(lora.alpha),
            lora_dropout=float(lora.dropout),
            lora_layers=function.normalize_lora_layers(getattr(lora, 'layers', None)),
            lora_target_modules='all-linear',
            hidden_size=int(scratch.hidden_size),
            num_layers=int(scratch.num_layers),
            num_heads=int(scratch.num_heads),
            dropout=float(scratch.dropout),
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
        )

    @property
    def run_id(self):
        is_llm = model_utils.match(self.model) is not None
        parts = [
            self.model,
            f'{self.repr_type}2{self.task_type}',
            f'bs{self.batch_size}',
            f'acc{self.accumulate_batch}' if self.accumulate_batch != 1 else None,
            'validonly' if self.valid_only else None,
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
        if self.task_type == 'sid' and self.code_beam_chunk_size != self.batch_size:
            parts.append(f'cbc{self.code_beam_chunk_size}')
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
        used_views = self.compile_config.used_views
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
        return payload
