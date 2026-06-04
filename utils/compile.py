import hashlib
import json
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


def short_config_hash(payload: dict, length: int = 8):
    serialized = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:length]


def compact_float(value):
    return f'{float(value):g}'


@dataclass
class CompileConfig:
    data: str
    model: str
    repr_type: Optional[str]
    repr_source_model: Optional[str]
    sid_export: Optional[str]
    sid_coder: Optional[str]
    hash_coder: Optional[str]
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
        self.repr_source_model = normalize_model_name(self.repr_source_model)
        self.sid_export = self.sid_export.lower() if self.sid_export else None
        self.sid_coder = str(self.sid_coder).strip().lower() if self.sid_coder else None
        self.hash_coder = str(self.hash_coder).strip().lower() if self.hash_coder else None

    @property
    def repr_types(self):
        return [part.strip().lower() for part in self.repr_type.split('+') if part.strip()]

    @property
    def sign_parts(self):
        used_views = set(self.repr_types + [self.task_type])
        uses_sid = 'sid' in used_views
        uses_hash = 'hash' in used_views
        uses_embedding = 'embedding' in used_views
        parts = [
            self.model,
            f'{self.repr_type}2{self.task_type}',
        ]
        if self.repr_combine != 'concat':
            parts.append(self.repr_combine)
        if self.maxitems != 0:
            parts.append(f'mi{self.maxitems}')
        if self.item_text_max_tokens != 50:
            parts.append(f'tl{self.item_text_max_tokens}')
        if self.model_max_length:
            parts.append(f'ml{self.model_max_length}')
        if self.repr_source_model and (uses_sid or uses_hash or uses_embedding):
            parts.append(f'rsm-{self.repr_source_model}')
        if self.sid_export and uses_sid:
            parts.append(f'se-{self.sid_export}')
        if self.sid_coder and uses_sid:
            parts.append(f'sc-{self.sid_coder}')
        if self.hash_coder and uses_hash:
            parts.append(f'hc-{self.hash_coder}')
        return parts

    @property
    def prepare_id(self):
        parts = self.sign_parts.copy()
        return '__'.join(parts)

    @property
    def config_dict(self):
        return asdict(self)
