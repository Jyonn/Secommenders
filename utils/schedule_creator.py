from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from utils.compile import CompileConfig, normalize_model_name
from utils.experiment_template import build_default_upstreams
from utils import model as model_utils

def _validate_compile_config(config: CompileConfig):
    supported_repr_types = {'uid', 'sid', 'hash', 'text', 'embedding'}
    supported_task_types = {'uid', 'sid', 'hash', 'embedding'}
    supported_repr_combines = {'concat', 'add'}

    repr_types = config.repr_types
    model_name = str(config.model).strip().lower()
    is_scratch_model = model_name == 'scratch'

    if not is_scratch_model and model_utils.match(model_name) is None:
        raise ValueError(
            f'Unknown model "{config.model}". '
            f'Use "scratch" for the scratch backbone or add the model alias to .model'
        )
    if not repr_types:
        raise ValueError('repr.type must contain at least one representation')
    if len(set(repr_types)) != len(repr_types):
        raise ValueError(f'repr.type contains duplicates: {config.repr_type}')
    unsupported_repr_types = [repr_type for repr_type in repr_types if repr_type not in supported_repr_types]
    if unsupported_repr_types:
        raise ValueError(f'Unsupported repr.type entries: {unsupported_repr_types}')
    if config.task_type not in supported_task_types:
        raise ValueError(f'Unsupported task.type: {config.task_type}')
    if config.repr_combine not in supported_repr_combines:
        raise ValueError(f'Unsupported repr.combine: {config.repr_combine}')
    if config.task_type not in repr_types:
        raise ValueError('repr.type must contain task.type so each item block starts with task representation')
    if repr_types[0] != config.task_type:
        raise ValueError('task.type must be the first entry in repr.type for causal mixed-view training')
    if config.repr_combine == 'add':
        if not (config.task_type == 'uid' and repr_types == ['uid', 'embedding']):
            raise ValueError(
                'repr.combine=add is currently only supported for uid+embedding history with uid targets'
            )
    if is_scratch_model and 'text' in repr_types:
        raise ValueError('scratch backbone currently does not support repr.type containing text')

    external_view_required = any(view in {'sid', 'hash', 'embedding'} for view in repr_types + [config.task_type])
    if external_view_required and not config.repr_source_model:
        raise ValueError('data.repr_source_model is required when repr.type or task.type uses sid/hash/embedding')
    if 'sid' in set(repr_types + [config.task_type]) and not config.sid_export:
        raise ValueError('data.sid_export is required when repr.type or task.type uses sid')
    if 'sid' in set(repr_types + [config.task_type]) and not config.sid_coder:
        raise ValueError('data.sid_coder is required when repr.type or task.type uses sid')
    if 'hash' in set(repr_types + [config.task_type]) and not config.hash_coder:
        raise ValueError('data.hash_coder is required when repr.type or task.type uses hash')


def _validate_job_runtime_args(args: dict[str, Any], compile_config: CompileConfig):
    valid_only = args.get('valid_only')
    test_only = bool(args.get('test_only', False))
    load_ckpt = args.get('load_ckpt')
    uid_decoding = str(args.get('uid_decoding', 'flat')).strip().lower()
    uid_cluster_levels = args.get('uid_cluster_levels')
    uid_cluster_topk = args.get('uid_cluster_topk')

    if valid_only and test_only:
        raise ValueError('trainer.valid_only and trainer.test_only cannot both be true')
    if test_only and not load_ckpt:
        raise ValueError('trainer.load_ckpt is required when trainer.test_only=true')
    if load_ckpt and not test_only:
        raise ValueError('trainer.load_ckpt is only supported together with trainer.test_only=true')
    if uid_decoding == 'hierarchical' and compile_config.task_type != 'uid':
        raise ValueError('trainer.uid_decoding=hierarchical is only supported when task_type=uid')
    if uid_decoding == 'hierarchical' and not uid_cluster_levels:
        raise ValueError('trainer.uid_cluster_levels is required when uid_decoding=hierarchical')
    if uid_decoding == 'hierarchical' and not uid_cluster_topk:
        raise ValueError('trainer.uid_cluster_topk is required when uid_decoding=hierarchical')


@dataclass
class _BackendConfig:
    host: str | None = None
    auth: str | None = None
    host_env: str | None = None
    auth_env: str | None = None

    def to_dict(self):
        payload = {}
        if self.host:
            payload['uri'] = self.host
        if self.auth:
            payload['auth_token'] = self.auth
        if self.host_env:
            payload['uri_env'] = self.host_env
        if self.auth_env:
            payload['auth_env'] = self.auth_env
        return payload


@dataclass(frozen=True)
class _UidVariant:
    decoding: str
    cluster_levels: str | None = None
    cluster_topk: str | None = None


@dataclass(frozen=True)
class _SidVariant:
    coder: str
    export: str


@dataclass(frozen=True)
class _StudySpec:
    name: str
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    targets: tuple[str, ...]
    histories: tuple[tuple[str, ...], ...]
    args: dict[str, Any]
    plan_fields: dict[str, Any]
    source_models: tuple[str, ...] | None = None
    sid_variants: tuple[_SidVariant, ...] | None = None
    hash_coders: tuple[str, ...] | None = None
    uid_variants: tuple[_UidVariant, ...] | None = None


def _ensure_tuple(values: str | Iterable[str] | None, *, lower: bool = True) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)
    normalized = []
    for value in items:
        item = str(value).strip()
        if not item:
            continue
        normalized.append(item.lower() if lower else item)
    return tuple(normalized)


def _normalize_history_entry(history: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(history, str):
        values = [history]
    else:
        values = list(history)
    normalized = []
    for value in values:
        item = str(value).strip().lower()
        if not item:
            continue
        if item in normalized:
            continue
        normalized.append(item)
    if not normalized:
        raise ValueError('history entry must contain at least one representation')
    return tuple(normalized)


def _normalize_histories(histories: str | Sequence[str] | Iterable[str | Sequence[str]]):
    if isinstance(histories, str):
        return (_normalize_history_entry(histories),)
    items = list(histories)
    if not items:
        return ()
    if all(isinstance(item, str) for item in items):
        return tuple(_normalize_history_entry(item) for item in items)
    return tuple(_normalize_history_entry(item) for item in items)


def _slugify(value: str):
    text = str(value).strip().lower()
    text = text.replace('+', 'p').replace(',', 'x').replace('/', '-')
    return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '-' for ch in text).strip('-')


def _normalize_sid_variant(value: Any) -> _SidVariant:
    if isinstance(value, _SidVariant):
        return value
    if isinstance(value, dict):
        coder = value.get('coder')
        export = value.get('export')
    else:
        try:
            coder, export = value
        except Exception as exc:
            raise ValueError('sid variant must be a (coder, export) pair or dict') from exc
    coder = str(coder).strip().lower()
    export = str(export).strip().lower()
    if not coder or not export:
        raise ValueError('sid variant requires both coder and export')
    return _SidVariant(coder=coder, export=export)


def _normalize_uid_variant(value: Any) -> _UidVariant:
    if isinstance(value, _UidVariant):
        return value
    if isinstance(value, str):
        decoding = value
        cluster_levels = None
        cluster_topk = None
    elif isinstance(value, dict):
        decoding = value.get('decoding')
        cluster_levels = value.get('cluster_levels')
        cluster_topk = value.get('cluster_topk')
    else:
        parts = list(value)
        if not parts:
            raise ValueError('uid variant must not be empty')
        decoding = parts[0]
        cluster_levels = parts[1] if len(parts) > 1 else None
        cluster_topk = parts[2] if len(parts) > 2 else None
    decoding = str(decoding).strip().lower()
    if not decoding:
        raise ValueError('uid variant requires decoding')
    cluster_levels = str(cluster_levels).strip() if cluster_levels is not None else None
    cluster_topk = str(cluster_topk).strip() if cluster_topk is not None else None
    return _UidVariant(
        decoding=decoding,
        cluster_levels=cluster_levels or None,
        cluster_topk=cluster_topk or None,
    )


class Job:
    def __init__(self, name: str):
        name = str(name).strip()
        if not name:
            raise ValueError('Job name is required')
        self.name = name
        self._args: dict[str, Any] = {}
        self._plan_fields: dict[str, Any] = {}
        self._repr_parts: list[str] | None = None

    def _set_arg(self, key: str, value: Any):
        self._args[key] = value
        return self

    def _set_string_arg(self, key: str, value: str, *, lower: bool = False):
        normalized = str(value).strip()
        if lower:
            normalized = normalized.lower()
        return self._set_arg(key, normalized)

    def _set_int_arg(self, key: str, value: int):
        return self._set_arg(key, int(value))

    def _set_float_arg(self, key: str, value: float):
        return self._set_arg(key, float(value))

    def clone(self, append_name: str | None = None, name: str | None = None):
        if name is not None and append_name is not None:
            raise ValueError('clone() accepts either append_name or name, not both')
        if name is not None:
            cloned_name = str(name).strip()
        elif append_name is not None:
            cloned_name = f'{self.name}{str(append_name)}'
        else:
            cloned_name = self.name
        cloned = Job(cloned_name)
        cloned._args = deepcopy(self._args)
        cloned._plan_fields = deepcopy(self._plan_fields)
        cloned._repr_parts = deepcopy(self._repr_parts)
        return cloned

    def data(self, value: str):
        return self._set_string_arg('data', value, lower=True)

    def model(self, value: str):
        return self._set_string_arg('model', value, lower=True)

    def task(self, value: str):
        return self._set_string_arg('task_type', value, lower=True)

    def repr(self, *repr_types: str):
        if not repr_types:
            raise ValueError('repr() expects at least one representation type')
        parts = []
        for repr_type in repr_types:
            value = str(repr_type).strip().lower()
            if not value:
                raise ValueError('repr() does not allow empty representation types')
            parts.append(value)
        self._repr_parts = parts
        self._args['repr_type'] = '+'.join(parts)
        return self

    def repr_source_model(self, value: str):
        return self._set_arg('repr_source_model', normalize_model_name(value))

    def sid_coder(self, value: str):
        return self._set_string_arg('sid_coder', value, lower=True)

    def sid_export(self, value: str):
        return self._set_string_arg('sid_export', value, lower=True)

    def hash_coder(self, value: str):
        return self._set_string_arg('hash_coder', value, lower=True)

    def repr_combine(self, value: str):
        return self._set_string_arg('repr_combine', value, lower=True)

    def maxitems(self, value: int):
        return self._set_int_arg('maxitems', value)

    def model_max_length(self, value: int):
        return self._set_int_arg('model_max_length', value)

    def item_text_max_tokens(self, value: int):
        return self._set_int_arg('item_text_max_tokens', value)

    def batch_size(self, value: int):
        return self._set_int_arg('batch_size', value)

    def accumulate_batch(self, value: int):
        return self._set_int_arg('accumulate_batch', value)

    def valid_only(self, value: bool | int):
        return self._set_arg('valid_only', value)

    def test_only(self, value: bool = True):
        return self._set_arg('test_only', bool(value))

    def load_ckpt(self, value: str):
        return self._set_string_arg('load_ckpt', value)

    def overwrite(self, value: str | bool = 'auto'):
        if isinstance(value, bool):
            value = 'true' if value else 'false'
        return self._set_arg('overwrite', str(value).strip().lower())

    def epochs(self, value: int):
        return self._set_int_arg('epochs', value)

    def learning_rate(self, value: float):
        return self._set_float_arg('learning_rate', value)

    def weight_decay(self, value: float):
        return self._set_float_arg('weight_decay', value)

    def seed(self, value: int):
        return self._set_int_arg('seed', value)

    def device(self, value: str):
        return self._set_string_arg('device', value)

    def num_gpus(self, value: int):
        return self._set_int_arg('num_gpus', value)

    def freeze_backbone(self, value: str | bool):
        return self._set_string_arg('freeze_backbone', value, lower=True)

    def uid_decoding(self, value: str):
        return self._set_string_arg('uid_decoding', value, lower=True)

    def uid_cluster_levels(self, value: str):
        return self._set_string_arg('uid_cluster_levels', value)

    def uid_cluster_topk(self, value: str):
        return self._set_string_arg('uid_cluster_topk', value)

    def code_decoding(self, value: str):
        return self._set_string_arg('code_decoding', value, lower=True)

    def main_metric(self, value: str):
        return self._set_string_arg('main_metric', value, lower=True)

    def metrics(self, *values: str):
        if len(values) == 1 and isinstance(values[0], str) and ',' in values[0]:
            metrics = [part.strip().lower() for part in values[0].split(',') if part.strip()]
        else:
            metrics = [str(value).strip().lower() for value in values if str(value).strip()]
        return self._set_arg('metrics', metrics)

    def patience(self, value: int):
        return self._set_int_arg('patience', value)

    def alignment(self, value: float):
        return self._set_float_arg('alignment', value)

    def code_beam_width(self, value: int):
        return self._set_int_arg('code_beam_width', value)

    def code_beam_chunk_size(self, value: int):
        return self._set_int_arg('code_beam_chunk_size', value)

    def code_collision_loss_weight(self, value: float):
        return self._set_float_arg('code_collision_loss_weight', value)

    def model_dtype(self, value: str):
        return self._set_string_arg('model_dtype', value, lower=True)

    def use_lora(self, value: str | bool):
        return self._set_string_arg('use_lora', value, lower=True)

    def lora_rank(self, value: int):
        return self._set_int_arg('lora_rank', value)

    def lora_alpha(self, value: int):
        return self._set_int_arg('lora_alpha', value)

    def lora_dropout(self, value: float):
        return self._set_float_arg('lora_dropout', value)

    def lora_layers(self, value: str):
        return self._set_string_arg('lora_layers', value)

    def lora_target_modules(self, value: str):
        return self._set_string_arg('lora_target_modules', value)

    def hidden_size(self, value: int):
        return self._set_int_arg('hidden_size', value)

    def num_layers(self, value: int):
        return self._set_int_arg('num_layers', value)

    def num_heads(self, value: int):
        return self._set_int_arg('num_heads', value)

    def dropout(self, value: float):
        return self._set_float_arg('dropout', value)

    def priority(self, value: int):
        self._plan_fields['priority'] = int(value)
        return self

    def batch_size_cap(self, value: int):
        self._plan_fields['batch_size_cap'] = int(value)
        return self

    def arg(self, key: str, value: Any):
        key = str(key).strip()
        if not key:
            raise ValueError('arg key must be non-empty')
        self._args[key] = value
        return self

    def args(self, **kwargs):
        for key, value in kwargs.items():
            self.arg(key, value)
        return self

    def _validated_args(self):
        required = ['data', 'model', 'task_type']
        missing = [key for key in required if not self._args.get(key)]
        if missing:
            raise ValueError(f'Job "{self.name}" is missing required args: {missing}')

        compile_config = CompileConfig(
            data=self._args['data'],
            model=self._args['model'],
            repr_type=self._args.get('repr_type'),
            repr_source_model=self._args.get('repr_source_model'),
            sid_export=self._args.get('sid_export'),
            sid_coder=self._args.get('sid_coder'),
            hash_coder=self._args.get('hash_coder'),
            task_type=self._args['task_type'],
            maxitems=int(self._args.get('maxitems', 20)),
            model_max_length=self._args.get('model_max_length'),
            item_text_max_tokens=int(self._args.get('item_text_max_tokens', 20)),
            repr_combine=self._args.get('repr_combine', 'concat'),
            upstreams=build_default_upstreams(self._args),
        )
        _validate_compile_config(compile_config)
        _validate_job_runtime_args(self._args, compile_config)

        payload = {key: value for key, value in self._args.items() if value is not None}
        payload['data'] = compile_config.data
        payload['model'] = compile_config.model
        payload['task_type'] = compile_config.task_type
        payload['repr_type'] = compile_config.repr_type
        if compile_config.repr_source_model:
            payload['repr_source_model'] = compile_config.repr_source_model
        if compile_config.sid_coder:
            payload['sid_coder'] = compile_config.sid_coder
        if compile_config.sid_export:
            payload['sid_export'] = compile_config.sid_export
        if compile_config.hash_coder:
            payload['hash_coder'] = compile_config.hash_coder
        if compile_config.repr_combine != 'concat' or 'repr_combine' in payload:
            payload['repr_combine'] = compile_config.repr_combine
        if 'uid_decoding' in payload:
            payload['uid_decoding'] = str(payload['uid_decoding']).strip().lower()
        if 'code_decoding' in payload:
            payload['code_decoding'] = str(payload['code_decoding']).strip().lower()
        if 'freeze_backbone' in payload:
            payload['freeze_backbone'] = str(payload['freeze_backbone']).strip().lower()
        if 'use_lora' in payload:
            payload['use_lora'] = str(payload['use_lora']).strip().lower()
        return payload

    def to_plan_item(self):
        payload = {
            'name': self.name,
            'args': self._validated_args(),
        }
        payload.update(self._plan_fields)
        return payload


class Schedule:
    def __init__(
        self,
        *,
        jobs: list[Job] | None = None,
        name: str | None = None,
        backend_host: str | None = None,
        backend_auth: str | None = None,
        backend_host_env: str | None = None,
        backend_auth_env: str | None = None,
        poll_interval_seconds: int = 15,
        effective_batch_size: int = 64,
    ):
        self.jobs = list(jobs or [])
        self.name = str(name).strip() if name else None
        self.poll_interval_seconds = int(poll_interval_seconds)
        self.effective_batch_size = int(effective_batch_size)
        self.backend = _BackendConfig(
            host=backend_host,
            auth=backend_auth,
            host_env=backend_host_env,
            auth_env=backend_auth_env,
        )
        self._default_args: dict[str, Any] = {}
        self._default_plan_fields: dict[str, Any] = {}
        self._default_source_models: tuple[str, ...] = ()
        self._default_sid_variants: tuple[_SidVariant, ...] = ()
        self._default_hash_coders: tuple[str, ...] = ()
        self._default_uid_variants: tuple[_UidVariant, ...] = ()
        self._studies: list[_StudySpec] = []
        self._validate_config()

    def _validate_config(self):
        if self.poll_interval_seconds <= 0:
            raise ValueError('poll_interval_seconds must be positive')
        if self.effective_batch_size <= 0:
            raise ValueError('effective_batch_size must be positive')
        if self.backend.auth and self.backend.auth_env:
            raise ValueError('Use either backend_auth or backend_auth_env, not both')
        if self.backend.host and self.backend.host_env:
            raise ValueError('Use either backend_host or backend_host_env, not both')

    def defaults(self, **kwargs):
        self._default_args.update(kwargs)
        return self

    def arg(self, key: str, value: Any):
        key = str(key).strip()
        if not key:
            raise ValueError('arg key must be non-empty')
        self._default_args[key] = value
        return self

    def args(self, **kwargs):
        return self.defaults(**kwargs)

    def plan_defaults(self, *, priority: int | None = None, batch_size_cap: int | None = None):
        if priority is not None:
            self._default_plan_fields['priority'] = int(priority)
        if batch_size_cap is not None:
            self._default_plan_fields['batch_size_cap'] = int(batch_size_cap)
        return self

    def main_metric(self, value: str):
        self._default_args['main_metric'] = str(value).strip().lower()
        return self

    def metrics(self, *values: str):
        if len(values) == 1 and isinstance(values[0], str) and ',' in values[0]:
            metrics = [part.strip().lower() for part in values[0].split(',') if part.strip()]
        else:
            metrics = [str(value).strip().lower() for value in values if str(value).strip()]
        self._default_args['metrics'] = metrics
        return self

    def priority(self, value: int):
        self._default_plan_fields['priority'] = int(value)
        return self

    def batch_size_cap(self, value: int):
        self._default_plan_fields['batch_size_cap'] = int(value)
        return self

    def source_models(self, *models: str):
        self._default_source_models = tuple(normalize_model_name(model) for model in models if str(model).strip())
        return self

    def sid_variants(self, *variants: Any):
        self._default_sid_variants = tuple(_normalize_sid_variant(variant) for variant in variants)
        return self

    def hash_coders(self, *coders: str):
        self._default_hash_coders = _ensure_tuple(coders)
        return self

    def uid_variants(self, *variants: Any):
        self._default_uid_variants = tuple(_normalize_uid_variant(variant) for variant in variants)
        return self

    def grid(
        self,
        name: str,
        *,
        datasets: str | Iterable[str],
        models: str | Iterable[str],
        targets: str | Iterable[str],
        histories: Iterable[str | Sequence[str]],
        args: dict[str, Any] | None = None,
        priority: int | None = None,
        batch_size_cap: int | None = None,
        source_models: str | Iterable[str] | None = None,
        sid_variants: Iterable[Any] | None = None,
        hash_coders: str | Iterable[str] | None = None,
        uid_variants: Iterable[Any] | None = None,
    ):
        histories_tuple = _normalize_histories(histories)
        if not histories_tuple:
            raise ValueError('grid() requires at least one history entry')
        spec = _StudySpec(
            name=str(name).strip(),
            datasets=_ensure_tuple(datasets),
            models=_ensure_tuple(models),
            targets=_ensure_tuple(targets),
            histories=histories_tuple,
            args=deepcopy(args or {}),
            plan_fields={
                key: value
                for key, value in {
                    'priority': int(priority) if priority is not None else None,
                    'batch_size_cap': int(batch_size_cap) if batch_size_cap is not None else None,
                }.items()
                if value is not None
            },
            source_models=(
                tuple(normalize_model_name(model) for model in _ensure_tuple(source_models))
                if source_models is not None else None
            ),
            sid_variants=(
                tuple(_normalize_sid_variant(variant) for variant in sid_variants)
                if sid_variants is not None else None
            ),
            hash_coders=_ensure_tuple(hash_coders) if hash_coders is not None else None,
            uid_variants=(
                tuple(_normalize_uid_variant(variant) for variant in uid_variants)
                if uid_variants is not None else None
            ),
        )
        if not spec.name:
            raise ValueError('grid() requires a non-empty name')
        if not spec.datasets:
            raise ValueError('grid() requires at least one dataset')
        if not spec.models:
            raise ValueError('grid() requires at least one model')
        if not spec.targets:
            raise ValueError('grid() requires at least one target')
        self._studies.append(spec)
        return self

    def _resolved_jobs(self):
        jobs = list(self.jobs)
        for spec in self._studies:
            jobs.extend(self._expand_study(spec))
        if not jobs:
            raise ValueError('Schedule requires at least one job or grid() study')
        return jobs

    def _job_name(self, spec: _StudySpec, dataset: str, model: str, history: tuple[str, ...], target: str, args: dict[str, Any]):
        history_label = '+'.join(history)
        parts = [spec.name, dataset, model, f'{history_label}2{target}']
        if 'repr_source_model' in args:
            parts.append(_slugify(args['repr_source_model']))
        if 'sid_coder' in args:
            parts.append(_slugify(args['sid_coder']))
        if 'sid_export' in args:
            parts.append(_slugify(args['sid_export']))
        if 'hash_coder' in args:
            parts.append(_slugify(args['hash_coder']))
        uid_decoding = args.get('uid_decoding')
        if uid_decoding:
            parts.append(_slugify(uid_decoding))
            if uid_decoding == 'hierarchical':
                if args.get('uid_cluster_levels'):
                    parts.append(f"lv{_slugify(args['uid_cluster_levels'])}")
                if args.get('uid_cluster_topk'):
                    parts.append(f"tk{_slugify(args['uid_cluster_topk'])}")
        return '_'.join(parts)

    def _semantic_source_options(self, spec: _StudySpec, *, requires_source: bool):
        if not requires_source:
            return [None]
        explicit = spec.source_models if spec.source_models is not None else self._default_source_models
        if explicit:
            return list(explicit)
        default_value = self._default_args.get('repr_source_model')
        if default_value:
            return [normalize_model_name(default_value)]
        raise ValueError('Semantic experiment requires source_models() or defaults(repr_source_model=...)')

    def _sid_options(self, spec: _StudySpec, *, uses_sid: bool):
        if not uses_sid:
            return [None]
        explicit = spec.sid_variants if spec.sid_variants is not None else self._default_sid_variants
        if explicit:
            return list(explicit)
        coder = self._default_args.get('sid_coder')
        export = self._default_args.get('sid_export')
        if coder and export:
            return [_SidVariant(coder=str(coder).strip().lower(), export=str(export).strip().lower())]
        raise ValueError('SID experiment requires sid_variants() or defaults(sid_coder=..., sid_export=...)')

    def _hash_options(self, spec: _StudySpec, *, uses_hash: bool):
        if not uses_hash:
            return [None]
        explicit = spec.hash_coders if spec.hash_coders is not None else self._default_hash_coders
        if explicit:
            return list(explicit)
        default_value = self._default_args.get('hash_coder')
        if default_value:
            return [str(default_value).strip().lower()]
        raise ValueError('Hash experiment requires hash_coders() or defaults(hash_coder=...)')

    def _uid_options(self, spec: _StudySpec, *, target: str):
        if target != 'uid':
            return [None]
        explicit = spec.uid_variants if spec.uid_variants is not None else self._default_uid_variants
        if explicit:
            return list(explicit)
        decoding = self._default_args.get('uid_decoding')
        if decoding:
            return [
                _UidVariant(
                    decoding=str(decoding).strip().lower(),
                    cluster_levels=self._default_args.get('uid_cluster_levels'),
                    cluster_topk=self._default_args.get('uid_cluster_topk'),
                )
            ]
        return [_UidVariant(decoding='flat')]

    def _expand_study(self, spec: _StudySpec):
        jobs = []
        for dataset in spec.datasets:
            for model in spec.models:
                for target in spec.targets:
                    for history in spec.histories:
                        repr_types = [target]
                        for view in history:
                            if view != target:
                                repr_types.append(view)
                        used_views = set(repr_types)
                        requires_source = any(view in {'sid', 'hash', 'embedding'} for view in used_views)
                        source_options = self._semantic_source_options(spec, requires_source=requires_source)
                        sid_options = self._sid_options(spec, uses_sid='sid' in used_views)
                        hash_options = self._hash_options(spec, uses_hash='hash' in used_views)
                        uid_options = self._uid_options(spec, target=target)
                        for source_model in source_options:
                            for sid_variant in sid_options:
                                for hash_coder in hash_options:
                                    for uid_variant in uid_options:
                                        args = deepcopy(self._default_args)
                                        args.update(spec.args)
                                        args['data'] = dataset
                                        args['model'] = model
                                        args['task_type'] = target
                                        args['repr_type'] = '+'.join(repr_types)
                                        if source_model is not None:
                                            args['repr_source_model'] = source_model
                                        if sid_variant is not None:
                                            args['sid_coder'] = sid_variant.coder
                                            args['sid_export'] = sid_variant.export
                                        if hash_coder is not None:
                                            args['hash_coder'] = hash_coder
                                        if uid_variant is not None:
                                            args['uid_decoding'] = uid_variant.decoding
                                            if uid_variant.cluster_levels is not None:
                                                args['uid_cluster_levels'] = uid_variant.cluster_levels
                                            if uid_variant.cluster_topk is not None:
                                                args['uid_cluster_topk'] = uid_variant.cluster_topk
                                        job = Job(self._job_name(spec, dataset, model, history, target, args))
                                        job.args(**args)
                                        plan_fields = deepcopy(self._default_plan_fields)
                                        plan_fields.update(spec.plan_fields)
                                        for key, value in plan_fields.items():
                                            if key == 'priority':
                                                job.priority(value)
                                            elif key == 'batch_size_cap':
                                                job.batch_size_cap(value)
                                        jobs.append(job)
        return jobs

    def _validate_jobs(self, jobs: list[Job]):
        seen = set()
        for job in jobs:
            if not isinstance(job, Job):
                raise TypeError('Schedule jobs must be Job instances')
            if job.name in seen:
                raise ValueError(f'Duplicate job name: {job.name}')
            seen.add(job.name)
            job.to_plan_item()

    def to_dict(self, path: str | Path | None = None):
        self._validate_config()
        jobs = self._resolved_jobs()
        self._validate_jobs(jobs)
        schedule_name = self.name
        if not schedule_name and path is not None:
            schedule_name = Path(path).stem
        payload = {
            'name': schedule_name or 'schedule',
            'poll_interval_seconds': self.poll_interval_seconds,
            'effective_batch_size': self.effective_batch_size,
            'experiments': [job.to_plan_item() for job in jobs],
        }
        backend_payload = self.backend.to_dict()
        if backend_payload:
            payload['backend'] = backend_payload
        return payload

    def export(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(path=path)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        return path
