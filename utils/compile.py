from dataclasses import asdict, dataclass
from typing import Optional


def normalize_model_name(name: Optional[str]):
    if not name:
        return None
    return name.replace('.', '').lower()


def canonicalize_repr_type(task_type: str, repr_type: Optional[str]):
    task = str(task_type).strip().lower()
    if not task:
        raise ValueError('task_type is required')

    parts = []
    if repr_type:
        parts = [part.strip().lower() for part in str(repr_type).split('+') if part.strip()]

    if task in parts:
        parts = [part for part in parts if part != task]
    parts = [task] + parts

    deduped = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return '+'.join(deduped)


@dataclass
class CompileConfig:
    data: str
    model: str
    repr_type: Optional[str]
    repr_model: Optional[str]
    repr_best: Optional[str]
    task_type: str
    maxitems: int
    model_max_length: Optional[int] = None
    item_text_max_tokens: int = 50
    repr_combine: str = 'concat'

    def __post_init__(self):
        self.data = str(self.data).lower()
        self.model = str(self.model).lower()
        self.task_type = str(self.task_type).lower()
        self.repr_type = canonicalize_repr_type(self.task_type, self.repr_type)
        self.repr_combine = str(self.repr_combine).lower()
        self.repr_model = normalize_model_name(self.repr_model)
        self.repr_best = self.repr_best.lower() if self.repr_best else None

    @property
    def repr_types(self):
        return [part.strip().lower() for part in self.repr_type.split('+') if part.strip()]

    @property
    def prepare_id(self):
        used_views = set(self.repr_types + [self.task_type])
        uses_sid = 'sid' in used_views
        uses_embedding = 'embedding' in used_views
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
        if self.repr_model and (uses_sid or uses_embedding):
            parts.append(f'reprmodel-{self.repr_model}')
        if self.repr_best and uses_sid:
            parts.append(f'reprbest-{self.repr_best}')
        return '__'.join(parts)

    @property
    def config_dict(self):
        return asdict(self)
