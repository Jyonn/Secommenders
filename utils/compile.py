import hashlib
import json
from dataclasses import asdict, dataclass
from copy import deepcopy
from typing import Optional


def normalize_model_name(name: Optional[str]):
    if not name:
        return None
    return name.replace('.', '').lower()


def canonicalize_repr_type(task_type: str, repr_type: Optional[str]):
    task_parts = canonicalize_task_type(task_type).split('+')
    if not task_parts:
        raise ValueError('task_type is required')

    parts = []
    if repr_type:
        parts = [part.strip().lower() for part in str(repr_type).split('+') if part.strip()]

    parts = [part for part in parts if part not in task_parts]
    parts = task_parts + parts

    deduped = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return '+'.join(deduped)


def canonicalize_task_type(task_type: str):
    parts = sorted({part.strip().lower() for part in str(task_type or '').split('+') if part.strip()})
    if not parts:
        raise ValueError('task_type is required')
    return '+'.join(parts)


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
    upstreams: Optional[dict] = None
    embedding: Optional[dict] = None
    representation_graph: Optional[dict] = None

    def __post_init__(self):
        self.data = str(self.data).lower()
        self.model = str(self.model).lower()
        self.task_type = canonicalize_task_type(self.task_type)
        self.repr_type = canonicalize_repr_type(self.task_type, self.repr_type)
        self.repr_combine = str(self.repr_combine).lower()
        self.repr_source_model = normalize_model_name(self.repr_source_model)
        self.sid_export = self.sid_export.lower() if self.sid_export else None
        self.sid_coder = str(self.sid_coder).strip().lower() if self.sid_coder else None
        self.hash_coder = str(self.hash_coder).strip().lower() if self.hash_coder else None
        self.upstreams = deepcopy(self.upstreams or {})
        self.embedding = deepcopy(self.embedding or {})
        self.representation_graph = deepcopy(self.representation_graph or {})

    @property
    def representation_names(self):
        if self.representation_graph:
            return list(self.representation_graph['encoder']['representations'])
        return self.repr_types

    @property
    def target_names(self):
        if self.representation_graph:
            return [target['representation'] for target in self.representation_graph['decoder']['targets']]
        return self.task_types

    def representation_kind(self, name):
        if self.representation_graph:
            return self.representation_graph['representations'][name]['type']
        return name

    def representation_spec(self, name):
        if self.representation_graph:
            return self.representation_graph['representations'][name]
        return {'type': name}

    def names_for_kind(self, kind, *, targets=False):
        names = self.target_names if targets else self.representation_names
        return [name for name in names if self.representation_kind(name) == kind]

    def primary_name(self, kind, *, targets=False):
        names = self.names_for_kind(kind, targets=targets)
        return names[0] if names else None

    def target_spec(self, name):
        if not self.representation_graph:
            return {'representation': name}
        for target in self.representation_graph['decoder']['targets']:
            if target['representation'] == name:
                return target
        return None

    def upstream_for(self, name):
        if name in self.upstreams:
            return deepcopy(self.upstreams[name])
        kind = self.representation_kind(name)
        return deepcopy(self.upstreams.get(kind) or {})

    @property
    def repr_types(self):
        return [part.strip().lower() for part in self.repr_type.split('+') if part.strip()]

    @property
    def task_types(self):
        return self.task_type.split('+')

    @property
    def used_views(self):
        return set(self.repr_types + self.task_types)

    @property
    def compile_upstreams(self):
        if self.representation_graph:
            upstreams = {}
            for name in self.representation_names:
                kind = self.representation_kind(name)
                if kind in {'sid', 'hash', 'uid'}:
                    upstream = self.upstream_for(name)
                    if upstream:
                        upstreams[name] = upstream
            return upstreams
        upstreams = {}
        if 'sid' in self.used_views and self.upstreams.get('sid'):
            upstreams['sid'] = deepcopy(self.upstreams['sid'])
        if 'hash' in self.used_views and self.upstreams.get('hash'):
            upstreams['hash'] = deepcopy(self.upstreams['hash'])
        if 'uid' in self.used_views and self.upstreams.get('uid'):
            upstreams['uid'] = deepcopy(self.upstreams['uid'])
        return upstreams

    @property
    def sign_parts(self):
        uses_sid = 'sid' in self.used_views
        uses_hash = 'hash' in self.used_views
        uses_embedding = 'embedding' in self.used_views
        parts = [
            self.model,
            (
                f'{"+".join(self.representation_names)}2{"+".join(self.target_names)}'
                if self.representation_graph
                else f'{self.repr_type}2{self.task_type}'
            ),
        ]
        if self.repr_combine != 'concat':
            parts.append(self.repr_combine)
        if self.maxitems != 0:
            parts.append(f'mi{self.maxitems}')
        if self.item_text_max_tokens != 50:
            parts.append(f'tl{self.item_text_max_tokens}')
        if self.model_max_length:
            parts.append(f'ml{self.model_max_length}')
        if self.representation_graph:
            parts.append(f'rg{short_config_hash(self.representation_graph)}')
        if not self.representation_graph and self.repr_source_model and (uses_sid or uses_hash or uses_embedding):
            parts.append(f'rsm-{self.repr_source_model}')
        if uses_embedding and self.embedding:
            parts.append(f'emb{short_config_hash(self.embedding)}')
        if self.sid_export and uses_sid:
            parts.append(f'se-{self.sid_export}')
        if self.sid_coder and uses_sid:
            parts.append(f'sc-{self.sid_coder}')
        if self.hash_coder and uses_hash:
            parts.append(f'hc-{self.hash_coder}')
        if not self.representation_graph:
            for name, upstream in sorted(self.compile_upstreams.items()):
                parts.append(f'{name}u{short_config_hash(upstream)}')
        return parts

    @property
    def prepare_id(self):
        parts = self.sign_parts.copy()
        return '__'.join(parts)

    @property
    def config_dict(self):
        payload = asdict(self)
        payload['upstreams'] = self.compile_upstreams
        if self.representation_graph:
            for key in ('repr_source_model', 'sid_export', 'sid_coder', 'hash_coder', 'upstreams', 'embedding'):
                payload.pop(key, None)
        if not any(view in {'sid', 'hash', 'embedding'} for view in self.used_views):
            payload.pop('repr_source_model', None)
        if 'embedding' not in self.used_views or not payload.get('embedding'):
            payload.pop('embedding', None)
        if 'sid' not in self.used_views:
            payload.pop('sid_export', None)
            payload.pop('sid_coder', None)
        if 'hash' not in self.used_views:
            payload.pop('hash_coder', None)
        if not payload.get('upstreams'):
            payload.pop('upstreams', None)
        if not payload.get('representation_graph'):
            payload.pop('representation_graph', None)
        return payload
