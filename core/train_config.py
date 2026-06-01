from dataclasses import dataclass
from typing import Optional

from utils import function
from utils import model as model_utils
from utils.compile import CompileConfig, canonicalize_repr_type, compact_float, normalize_model_name, short_config_hash


@dataclass
class TrainConfig:
    data: str
    model: str
    repr_type: str
    repr_model: Optional[str]
    repr_best: Optional[str]
    repr_combine: str
    task_type: str
    maxitems: int
    model_max_length: Optional[int]
    item_text_max_tokens: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: Optional[str]
    num_gpus: int
    freeze_backbone: str
    main_metric: str
    metrics: list[str]
    patience: int
    alignment_enable: bool
    alignment_weight: float
    sid_beam_width: int
    sid_collision_loss_weight: float
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
        alignment = trainer.alignment
        model = configurations.config.model
        lora = model.lora
        scratch = model.config
        try:
            raw_repr_type = data_config.repr_type
        except Exception:
            raw_repr_type = None
        raw_task_type = str(data_config.task_type).lower()
        normalized_repr_type = canonicalize_repr_type(raw_task_type, raw_repr_type)
        alignment_enable = alignment.enable if isinstance(alignment.enable, bool) else str(alignment.enable).lower() == 'true'
        raw_metrics = getattr(evaluator, 'metrics', [])
        if isinstance(raw_metrics, str):
            metrics = [metric.strip().lower() for metric in raw_metrics.split(',') if metric.strip()]
        else:
            metrics = [str(metric).strip().lower() for metric in list(raw_metrics) if str(metric).strip()]
        return cls(
            data=data_config.name.lower(),
            model=model.name.lower(),
            repr_type=normalized_repr_type,
            repr_model=normalize_model_name(data_config.repr_model),
            repr_best=data_config.repr_best.lower() if data_config.repr_best else None,
            repr_combine=data_config.repr_combine.lower(),
            task_type=raw_task_type,
            maxitems=int(data_config.maxitems),
            model_max_length=int(model.max_length) or None,
            item_text_max_tokens=int(data_config.item_text_max_tokens),
            batch_size=int(trainer.batch_size),
            epochs=int(trainer.epochs),
            learning_rate=float(trainer.learning_rate),
            weight_decay=float(trainer.weight_decay),
            seed=int(trainer.seed),
            device=trainer.device,
            num_gpus=int(getattr(trainer, 'num_gpus', 1)),
            freeze_backbone=str(trainer.freeze_backbone).lower(),
            main_metric=str(getattr(evaluator, 'main_metric', 'loss')).strip().lower(),
            metrics=metrics,
            patience=int(evaluator.patience),
            alignment_enable=alignment_enable,
            alignment_weight=float(alignment.weight),
            sid_beam_width=int(getattr(trainer, 'sid_beam_width', 20)),
            sid_collision_loss_weight=float(getattr(trainer, 'sid_collision_loss_weight', 0.1)),
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
            repr_model=self.repr_model,
            repr_best=self.repr_best,
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
            f'lr{compact_float(self.learning_rate)}',
            f'wd{compact_float(self.weight_decay)}',
        ]
        if self.repr_combine != 'concat':
            parts.append(self.repr_combine)
        if self.freeze_backbone != 'auto':
            parts.append(f'fr-{self.freeze_backbone}')
        if self.use_lora != 'auto':
            parts.append(f'lo-{self.use_lora}')
        if self.alignment_enable:
            parts.append(f'al{compact_float(self.alignment_weight)}')
        if self.task_type == 'sid' and self.sid_beam_width != 20:
            parts.append(f'bm{self.sid_beam_width}')
        if self.task_type == 'sid' and self.sid_collision_loss_weight != 0.1:
            parts.append(f'scw{compact_float(self.sid_collision_loss_weight)}')
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
        parts.append(f'h{short_config_hash(self.__dict__)}')
        return '__'.join(parts)
