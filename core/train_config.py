from dataclasses import dataclass
from typing import Optional

from utils.compile import CompileConfig, normalize_model_name


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
    freeze_backbone: str
    patience: int
    use_lora: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float

    @classmethod
    def from_refconfig(cls, configurations):
        data_config = configurations.config.data
        trainer = configurations.config.trainer
        model = configurations.config.model
        lora = model.lora
        scratch = model.config
        return cls(
            data=data_config.name.lower(),
            model=model.name.lower(),
            repr_type=data_config.repr_type.lower(),
            repr_model=normalize_model_name(data_config.repr_model),
            repr_best=data_config.repr_best.lower() if data_config.repr_best else None,
            repr_combine=data_config.repr_combine.lower(),
            task_type=data_config.task_type.lower(),
            maxitems=int(data_config.maxitems),
            model_max_length=int(model.max_length) or None,
            item_text_max_tokens=int(data_config.item_text_max_tokens),
            batch_size=int(trainer.batch_size),
            epochs=int(trainer.epochs),
            learning_rate=float(trainer.learning_rate),
            weight_decay=float(trainer.weight_decay),
            seed=int(trainer.seed),
            device=trainer.device,
            freeze_backbone=str(trainer.freeze_backbone).lower(),
            patience=int(trainer.patience),
            use_lora=str(lora.use).lower(),
            lora_rank=int(lora.rank),
            lora_alpha=int(lora.alpha),
            lora_dropout=float(lora.dropout),
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
        parts = [
            self.compile_config.prepare_id,
            f'bs-{self.batch_size}',
            f'lr-{self.learning_rate:g}',
            f'wd-{self.weight_decay:g}',
            f'seed-{self.seed}',
            f'freeze-{self.freeze_backbone}',
            f'lora-{self.use_lora}',
        ]
        if self.model != 'transformer':
            parts.extend(
                [
                    f'r-{self.lora_rank}',
                    f'a-{self.lora_alpha}',
                    f'drop-{self.lora_dropout:g}',
                ]
            )
        return '__'.join(parts)
