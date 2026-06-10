from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from utils.compile import CompileConfig, normalize_model_name
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


class Job:
    def __init__(self, name: str):
        name = str(name).strip()
        if not name:
            raise ValueError('Job name is required')
        self.name = name
        self._args: dict[str, Any] = {}
        self._plan_fields: dict[str, Any] = {}
        self._repr_parts: list[str] | None = None

    def data(self, value: str):
        self._args['data'] = str(value).strip().lower()
        return self

    def model(self, value: str):
        self._args['model'] = str(value).strip().lower()
        return self

    def task(self, value: str):
        self._args['task_type'] = str(value).strip().lower()
        return self

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
        self._args['repr_source_model'] = normalize_model_name(value)
        return self

    def sid_coder(self, value: str):
        self._args['sid_coder'] = str(value).strip().lower()
        return self

    def sid_export(self, value: str):
        self._args['sid_export'] = str(value).strip().lower()
        return self

    def hash_coder(self, value: str):
        self._args['hash_coder'] = str(value).strip().lower()
        return self

    def repr_combine(self, value: str):
        self._args['repr_combine'] = str(value).strip().lower()
        return self

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
        jobs: list[Job],
        name: str | None = None,
        backend_host: str | None = None,
        backend_auth: str | None = None,
        backend_host_env: str | None = None,
        backend_auth_env: str | None = None,
        poll_interval_seconds: int = 15,
        effective_batch_size: int = 64,
    ):
        if not jobs:
            raise ValueError('Schedule requires at least one job')
        self.jobs = jobs
        self.name = str(name).strip() if name else None
        self.poll_interval_seconds = int(poll_interval_seconds)
        self.effective_batch_size = int(effective_batch_size)
        self.backend = _BackendConfig(
            host=backend_host,
            auth=backend_auth,
            host_env=backend_host_env,
            auth_env=backend_auth_env,
        )
        self._validate()

    def _validate(self):
        if self.poll_interval_seconds <= 0:
            raise ValueError('poll_interval_seconds must be positive')
        if self.effective_batch_size <= 0:
            raise ValueError('effective_batch_size must be positive')
        if self.backend.auth and self.backend.auth_env:
            raise ValueError('Use either backend_auth or backend_auth_env, not both')
        if self.backend.host and self.backend.host_env:
            raise ValueError('Use either backend_host or backend_host_env, not both')
        seen = set()
        for job in self.jobs:
            if not isinstance(job, Job):
                raise TypeError('Schedule jobs must be Job instances')
            if job.name in seen:
                raise ValueError(f'Duplicate job name: {job.name}')
            seen.add(job.name)
            job.to_plan_item()

    def to_dict(self, path: str | Path | None = None):
        schedule_name = self.name
        if not schedule_name and path is not None:
            schedule_name = Path(path).stem
        payload = {
            'name': schedule_name or 'schedule',
            'poll_interval_seconds': self.poll_interval_seconds,
            'effective_batch_size': self.effective_batch_size,
            'experiments': [job.to_plan_item() for job in self.jobs],
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
