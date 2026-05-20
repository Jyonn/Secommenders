from dataclasses import asdict, dataclass
from typing import Optional


def normalize_model_name(name: Optional[str]):
    if not name:
        return None
    return name.replace('.', '').lower()


@dataclass
class CompileConfig:
    data: str
    model: str
    repr_type: str
    repr_model: Optional[str]
    repr_best: Optional[str]
    task_type: str
    maxitems: int
    model_max_length: Optional[int] = None
    item_text_max_tokens: int = 50
    repr_combine: str = 'concat'

    @property
    def repr_types(self):
        return [part.strip().lower() for part in self.repr_type.split('+') if part.strip()]

    @property
    def prepare_id(self):
        parts = [
            f'model-{self.model}',
            f'repr-{self.repr_type}',
            f'combine-{self.repr_combine}',
            f'task-{self.task_type}',
            f'maxitems-{"auto" if self.maxitems == 0 else self.maxitems}',
            f'textlen-{self.item_text_max_tokens}',
        ]
        if self.model_max_length:
            parts.append(f'modellen-{self.model_max_length}')
        if self.repr_model:
            parts.append(f'reprmodel-{self.repr_model}')
        if self.repr_best:
            parts.append(f'reprbest-{self.repr_best}')
        return '__'.join(parts)

    @property
    def config_dict(self):
        return asdict(self)
