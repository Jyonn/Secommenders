import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from notificator import Notificator

from core.train_config import TrainConfig
from utils.compile import short_config_hash
from utils.config_init import ConfigInit
from utils.server import Server


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / 'artifacts' / 'scheduler'
BATCH_LADDER = [64, 32, 16, 8, 4, 2, 1]
MODEL_BATCH_CAPS = {
    'scratch': 64,
    'qwen35th08b': 32,
    'qwen35th4b': 16,
    'llama3': 4,
    'qwen35th9b': 4,
}
MODEL_FREE_MEMORY_REQUIREMENTS_MB = {
    'scratch': 10_000,
    'qwen35th08b': 20_000,
    'qwen35th4b': 40_000,
    'llama3': 80_000,
    'qwen35th9b': 80_000,
}
OOM_PATTERNS = [
    'cuda out of memory',
    'outofmemoryerror',
    'torch.cuda.outofmemoryerror',
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(text: str):
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', text).strip('._-') or 'exp'


def coerce_cli_value(value: str):
    lowered = value.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered == 'null':
        return None
    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_trainer_command(command: str):
    tokens = shlex.split(command)
    trainer_index = None
    for index, token in enumerate(tokens):
        if token.endswith('trainer.py'):
            trainer_index = index
            break
    if trainer_index is None:
        raise ValueError(f'Command does not reference trainer.py: {command}')

    args = {}
    key = None
    for token in tokens[trainer_index + 1:]:
        if token in {'>', '>>', '2>', '2>>', '&', '2>&1'}:
            break
        if key is not None:
            args[key] = coerce_cli_value(token)
            key = None
            continue
        if not token.startswith('--'):
            continue
        key = token[2:]
    if key is not None:
        raise ValueError(f'Missing value for argument --{key} in command: {command}')
    return args


def build_train_config(arg_map: dict):
    kwargs = dict(arg_map)
    kwargs['config'] = 'config/trainer.yaml'
    configurations = ConfigInit(
        required_args=[],
        default_args=dict(config='config/trainer.yaml'),
        makedirs=[],
    ).parse_kwargs(kwargs)
    return TrainConfig.from_refconfig(configurations)


def effective_batch_to_accumulate(batch_size: int, effective_batch_size: int):
    if effective_batch_size % batch_size != 0:
        raise ValueError(
            f'effective_batch_size={effective_batch_size} must be divisible by batch_size={batch_size}'
        )
    return effective_batch_size // batch_size


def initial_batch_cap(model_name: str):
    key = str(model_name).lower()
    if key not in MODEL_BATCH_CAPS:
        raise ValueError(
            f'Unsupported scheduler model "{model_name}". '
            f'Supported models: {sorted(MODEL_BATCH_CAPS)}'
        )
    return MODEL_BATCH_CAPS[key]


def uses_embedding_path(base_args: dict):
    repr_type = str(base_args.get('repr_type') or '').lower()
    task_type = str(base_args.get('task_type') or '').lower()
    repr_parts = [part.strip() for part in repr_type.split('+') if part.strip()]
    return 'embedding' in repr_parts or task_type == 'embedding'


def required_free_memory_mb(base_args: dict):
    model_name = str(base_args.get('model') or '').lower()
    if model_name not in MODEL_FREE_MEMORY_REQUIREMENTS_MB:
        raise ValueError(
            f'Unsupported scheduler model "{model_name}". '
            f'Supported models: {sorted(MODEL_FREE_MEMORY_REQUIREMENTS_MB)}'
        )
    base_requirement = MODEL_FREE_MEMORY_REQUIREMENTS_MB[model_name]
    if not uses_embedding_path(base_args):
        return base_requirement

    if model_name == 'scratch':
        return 20_000
    if model_name == 'qwen35th08b':
        return 40_000
    return 80_000


def next_smaller_batch_size(current_batch_size: int):
    for candidate in BATCH_LADDER:
        if candidate < current_batch_size:
            return candidate
    return None


def query_gpus():
    command = [
        'nvidia-smi',
        '--query-gpu=index,memory.free,memory.total',
        '--format=csv,noheader,nounits',
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    gpus = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        index_text, free_text, total_text = [part.strip() for part in line.split(',')]
        gpus.append(
            {
                'index': int(index_text),
                'free_mb': int(free_text),
                'total_mb': int(total_text),
            }
        )
    return sorted(gpus, key=lambda item: item['free_mb'], reverse=True)


def read_log_tail(log_path: Path, limit_bytes: int = 200_000):
    if not log_path.exists():
        return ''
    data = log_path.read_bytes()
    return data[-limit_bytes:].decode('utf-8', errors='ignore')


def log_has_oom(log_tail: str):
    lowered = log_tail.lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


def log_reached_test(log_tail: str):
    markers = [
        'loaded best checkpoint',
        'loaded checkpoint ',
        'test:',
        'best_epoch=',
        'start test-only evaluation',
    ]
    lowered = log_tail.lower()
    return any(marker in lowered for marker in markers)


def build_args_for_phase(base_args: dict, *, batch_size: int, effective_batch_size: int, phase: str, load_ckpt=None):
    args = deepcopy(base_args)
    args['batch_size'] = int(batch_size)
    args['accumulate_batch'] = int(effective_batch_to_accumulate(batch_size, effective_batch_size))
    args['code_beam_chunk_size'] = int(batch_size)
    args.pop('valid_only', None)
    args.pop('test_only', None)
    args.pop('load_ckpt', None)
    if phase == 'precheck':
        args['valid_only'] = 1
    elif phase == 'test':
        args['test_only'] = True
        args['load_ckpt'] = str(load_ckpt)
    return args


def trainer_command_from_args(args: dict):
    command = [sys.executable, 'trainer.py']
    for key, value in args.items():
        if value is None:
            continue
        command.extend([f'--{key}', str(value).lower() if isinstance(value, bool) else str(value)])
    return command


def run_dir_for_args(args: dict):
    config = build_train_config(args)
    return ROOT / 'artifacts' / 'trained' / config.data / config.run_id


def run_dir_completed(run_dir: Path):
    meta_path = run_dir / 'meta.json'
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    return isinstance(meta, dict) and 'test_metrics' in meta


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def looks_like_tqdm_progress(line: str):
    text = str(line).strip()
    if not text:
        return False
    if text.count('|') < 2:
        return False
    return any(token in text for token in ('%|', 'it/s', 's/it'))


def collapse_tqdm_progress(text: str):
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    collapsed = []
    pending_progress = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if looks_like_tqdm_progress(line):
            pending_progress = line
            continue
        if pending_progress is not None:
            collapsed.append(pending_progress)
            pending_progress = None
        collapsed.append(line)
    if pending_progress is not None:
        collapsed.append(pending_progress)
    return '\n'.join(collapsed)


def read_log_for_report(path: Path, max_bytes: int = 2_000_000):
    if not path.exists():
        return ''
    data = path.read_bytes()
    text = collapse_tqdm_progress(data.decode('utf-8', errors='ignore'))
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode('utf-8', errors='ignore')


def performance_from_meta(meta: dict | None):
    if not isinstance(meta, dict):
        return None
    for key in ('test_metrics', 'valid_metrics', 'checkpoint_valid_metrics'):
        metrics = meta.get(key)
        if isinstance(metrics, dict):
            return metrics
    return None


class SchedulerNotifier:
    def __init__(
        self,
        *,
        client,
        bark: str,
        sound: str | None = None,
        icon: str | None = None,
        group: str | None = None,
        url: str | None = None,
        title_prefix: str = 'Secommenders',
    ):
        self.client = client
        self.bark = bark
        self.sound = sound
        self.icon = icon
        self.group = group
        self.url = url
        self.title_prefix = title_prefix

    @classmethod
    def from_config(cls, conf: dict | None):
        if not conf:
            return None

        name = str(conf.get('name') or '').strip()
        token = str(conf.get('token') or os.environ.get(conf.get('token_env', ''), '')).strip()
        bark = str(conf.get('bark') or os.environ.get(conf.get('bark_env', ''), '')).strip()
        host = str(conf.get('host') or os.environ.get(conf.get('host_env', ''), '')).strip() or None
        locale = str(conf.get('locale') or '').strip() or None
        if not name or not token or not bark:
            print(
                'warning: notificator disabled because name/token/bark is incomplete '
                f'(name={bool(name)} token={bool(token)} bark={bool(bark)})'
            )
            return None
        client = Notificator(name=name, token=token, host=host, locale=locale) if locale else Notificator(
            name=name,
            token=token,
            host=host,
        )
        return cls(
            client=client,
            bark=bark,
            sound=str(conf.get('sound') or '').strip() or None,
            icon=str(conf.get('icon') or '').strip() or None,
            group=str(conf.get('group') or '').strip() or None,
            url=str(conf.get('url') or '').strip() or None,
            title_prefix=str(conf.get('title_prefix') or 'Secommenders').strip() or 'Secommenders',
        )

    def send(self, title: str, body: str):
        final_title = f'{self.title_prefix}: {title}'.strip()
        try:
            self.client.bark(
                self.bark,
                'text',
                body,
                title=final_title,
                sound=self.sound,
                icon=self.icon,
                group=self.group,
                url=self.url,
            )
            return True
        except Exception as exc:
            print(f'warning: failed to send notificator message "{final_title}": {repr(exc)}')
            return False


def needs_oom_precheck(base_args: dict):
    task_type = str(base_args.get('task_type', '')).lower()
    return task_type in {'sid', 'hash'}


class Scheduler:
    def __init__(self, plan_path: Path):
        self.plan_path = Path(plan_path)
        self.plan = yaml.safe_load(self.plan_path.read_text())
        self.plan_name = sanitize_name(self.plan.get('name') or self.plan_path.stem)
        self.poll_interval = int(self.plan.get('poll_interval_seconds', 15))
        self.effective_batch_size = int(self.plan.get('effective_batch_size', 64))
        self.output_dir = ARTIFACT_ROOT / self.plan_name
        self.logs_dir = self.output_dir / 'logs'
        self.state_path = self.output_dir / 'state.json'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.server = self._build_server(self.plan.get('backend') or {})
        self.notifier = SchedulerNotifier.from_config(self.plan.get('notificator') or {})
        self.experiments = self._load_or_initialize_state()
        self.active_jobs = {}

    @staticmethod
    def _build_server(backend_conf: dict):
        if not backend_conf:
            return Server.from_env()

        uri = backend_conf.get('uri')
        auth = backend_conf.get('auth_token')
        uri_env = backend_conf.get('uri_env')
        auth_env = backend_conf.get('auth_env')
        if uri_env:
            uri = os.environ.get(uri_env, uri)
        if auth_env:
            auth = os.environ.get(auth_env, auth)
        uri = uri or os.environ.get(Server.ENV_URI)
        auth = auth or os.environ.get(Server.ENV_AUTH)
        if not uri or not auth:
            if backend_conf:
                print(
                    'warning: backend reporting disabled because backend config is incomplete '
                    f'(uri={"set" if uri else "missing"}, auth={"set" if auth else "missing"}, '
                    f'uri_env={uri_env!r}, auth_env={auth_env!r})'
                )
            return None
        return Server(uri=uri, auth=auth)

    def _build_remote_spec(self, raw_exp: dict, index: int, base_args: dict, batch_size: int):
        logical_args = build_args_for_phase(
            base_args,
            batch_size=batch_size,
            effective_batch_size=self.effective_batch_size,
            phase='train',
        )
        config = build_train_config(logical_args)
        command = raw_exp.get('command')
        if command:
            report_command = ' '.join(command.strip().split())
        else:
            report_command = ' '.join(shlex.quote(part) for part in trainer_command_from_args(logical_args))
        payload = {
            'plan_name': self.plan_name,
            'plan_path': str(self.plan_path),
            'experiment_index': index,
            'experiment_name': raw_exp.get('name') or f'exp{index:03d}',
            'effective_batch_size': self.effective_batch_size,
            'base_args': base_args,
            'logical_train_args': logical_args,
            'run_id': config.run_id,
            'compile_prepare_id': config.compile_config.prepare_id,
        }
        signature = short_config_hash(payload, length=16)
        seed = int(base_args.get('seed', getattr(config, 'seed', 42)))
        return {
            'report_signature': signature,
            'report_command': report_command,
            'report_configuration': json.dumps(payload, indent=2, sort_keys=True),
            'report_seed': seed,
            'report_session': None,
            'report_uploaded_at': None,
            'report_upload_error': None,
        }

    def _normalize_experiment(self, raw_exp: dict, index: int):
        if 'args' in raw_exp:
            base_args = deepcopy(raw_exp['args'])
        elif 'command' in raw_exp:
            base_args = parse_trainer_command(raw_exp['command'])
        else:
            raise ValueError(f'Experiment #{index} must provide either "args" or "command"')

        if 'data' not in base_args or 'model' not in base_args or 'task_type' not in base_args:
            raise ValueError(f'Experiment #{index} is missing one of required args: data/model/task_type')

        name = sanitize_name(raw_exp.get('name') or f'exp{index:03d}')
        batch_cap = int(raw_exp.get('batch_size_cap') or initial_batch_cap(str(base_args['model'])))
        batch_size = next((candidate for candidate in BATCH_LADDER if candidate <= batch_cap), None)
        if batch_size is None:
            raise ValueError(f'No supported batch size found for experiment {name} with cap={batch_cap}')
        phase = 'precheck' if needs_oom_precheck(base_args) else 'train'
        run_dir = run_dir_for_args(
            build_args_for_phase(
                base_args,
                batch_size=batch_size,
                effective_batch_size=self.effective_batch_size,
                phase=phase,
            )
        )
        status = 'done' if run_dir_completed(run_dir) else 'pending'
        exp = {
            'name': name,
            'priority': int(raw_exp.get('priority', 0)),
            'base_args': base_args,
            'batch_size_cap': batch_cap,
            'required_free_memory_mb': int(raw_exp.get('required_free_memory_mb') or required_free_memory_mb(base_args)),
            'batch_size': batch_size,
            'phase': phase,
            'status': status,
            'retries': 0,
            'test_retries': 0,
            'run_dir': str(run_dir) if status == 'done' else None,
            'ckpt_path': None,
            'log_path': None,
            'last_error': 'skipped_existing_run' if status == 'done' else None,
            'started_at': None,
            'finished_at': utc_now_iso() if status == 'done' else None,
            'notification_marks': {},
        }
        exp.update(self._build_remote_spec(raw_exp, index, base_args, batch_size))
        return exp

    def _migrate_experiment_state(self, exp: dict):
        base_args = exp.get('base_args') or {}
        if not base_args:
            return exp
        exp.setdefault('batch_size_cap', initial_batch_cap(str(base_args.get('model'))))
        exp.setdefault('required_free_memory_mb', required_free_memory_mb(base_args))
        exp.setdefault('phase', 'precheck' if needs_oom_precheck(base_args) else 'train')
        exp.setdefault('retries', 0)
        exp.setdefault('test_retries', 0)
        exp.setdefault('run_dir', None)
        exp.setdefault('ckpt_path', None)
        exp.setdefault('log_path', None)
        exp.setdefault('last_error', None)
        exp.setdefault('started_at', None)
        exp.setdefault('finished_at', None)
        exp.setdefault('notification_marks', {})
        exp.setdefault('report_session', None)
        exp.setdefault('report_uploaded_at', None)
        exp.setdefault('report_upload_error', None)
        return exp

    def _load_or_initialize_state(self):
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            experiments = state.get('experiments', [])
            for exp in experiments:
                exp = self._migrate_experiment_state(exp)
                if exp.get('status') == 'done' and exp.get('run_dir') and run_dir_completed(Path(exp['run_dir'])):
                    continue
                if exp.get('status') == 'done' and exp.get('run_dir') and not run_dir_completed(Path(exp['run_dir'])):
                    exp['status'] = 'pending'
                    exp['last_error'] = 'existing done state no longer has completed run_dir'
                if exp.get('status') == 'running':
                    exp['status'] = 'pending'
                    exp['last_error'] = 'scheduler restarted while job was running'
            return experiments

        raw_experiments = self.plan.get('experiments') or []
        if not raw_experiments:
            raise ValueError(f'No experiments found in plan {self.plan_path}')
        experiments = [self._normalize_experiment(raw_exp, index) for index, raw_exp in enumerate(raw_experiments, start=1)]
        self._save_state(experiments)
        return experiments

    @staticmethod
    def _notification_marked(exp: dict, key: str):
        return bool((exp.get('notification_marks') or {}).get(key))

    @staticmethod
    def _mark_notification(exp: dict, key: str):
        marks = exp.setdefault('notification_marks', {})
        marks[key] = utc_now_iso()

    @staticmethod
    def _metrics_summary(performance: dict | None):
        if not isinstance(performance, dict):
            return ''
        items = []
        for key in sorted(performance):
            value = performance[key]
            if isinstance(value, (int, float)):
                items.append(f'{key}={value:.4f}')
            else:
                items.append(f'{key}={value}')
        return ', '.join(items[:6])

    def _notify(self, exp: dict, key: str, title: str, body: str):
        if self.notifier is None or self._notification_marked(exp, key):
            return False
        if self.notifier.send(title, body):
            self._mark_notification(exp, key)
            return True
        return False

    def _save_state(self, experiments=None):
        payload = {
            'name': self.plan_name,
            'plan_path': str(self.plan_path),
            'effective_batch_size': self.effective_batch_size,
            'poll_interval_seconds': self.poll_interval,
            'updated_at': utc_now_iso(),
            'experiments': experiments if experiments is not None else self.experiments,
        }
        self.state_path.write_text(json.dumps(payload, indent=2) + '\n')

    def _pending_experiments(self):
        return sorted(
            [exp for exp in self.experiments if exp['status'] == 'pending'],
            key=lambda exp: (-exp['priority'], exp['name']),
        )

    def _terminal(self, exp: dict):
        return exp['status'] in {'done', 'failed'}

    def _all_terminal(self):
        return all(self._terminal(exp) for exp in self.experiments)

    def _needs_remote_sync(self, exp: dict):
        if self.server is None:
            return False
        status = exp.get('status')
        if status not in {'done', 'failed'}:
            return False
        if status == 'done':
            run_dir = exp.get('run_dir')
            if not run_dir or not run_dir_completed(Path(run_dir)):
                return False
        elif exp.get('phase') == 'precheck':
            return False
        return not exp.get('report_uploaded_at')

    def _has_unsynced_completed_experiments(self):
        for exp in self.experiments:
            if not self._needs_remote_sync(exp):
                continue
            return True
        return False

    def _sync_terminal_experiment(self, exp: dict):
        run_dir_text = exp.get('run_dir')
        run_dir = Path(run_dir_text) if run_dir_text else None
        meta = read_json_if_exists(run_dir / 'meta.json') if run_dir else None
        meta = meta or {}
        log_path = Path(exp['log_path']) if exp.get('log_path') else (run_dir / 'train.log' if run_dir else None)
        log_text = read_log_for_report(log_path) if log_path else ''
        performance = performance_from_meta(meta)
        if len(log_text.encode('utf-8')) >= 2_000_000:
            meta = dict(meta)
            meta['report_log_truncated'] = True

        session = self._ensure_remote_session(exp)
        register_reply = self.server.register_experiment(
            session,
            pid=meta.get('pid'),
            hostname=str(meta.get('hostname') or ''),
            run_dir=str(run_dir) if run_dir else '',
            log_path=str(meta.get('log_path') or log_path or ''),
            command=str(meta.get('command') or exp.get('report_command') or ''),
            phase=str(exp.get('phase') or ''),
        )
        if not register_reply.ok:
            raise ValueError(
                f'failed to register remote experiment: {register_reply.msg or register_reply.identifier}'
            )
        remote_status = 'completed' if exp.get('status') == 'done' else 'failed'
        error_text = ''
        if remote_status == 'failed':
            error_text = str(meta.get('error') or exp.get('last_error') or '')
        reply = self.server.update_experiment(
            session,
            status=remote_status,
            phase=str(exp.get('phase') or ''),
            meta=meta,
            performance=performance,
            log=log_text,
            error=error_text,
        )
        if not reply.ok:
            raise ValueError(f'failed to update remote experiment: {reply.msg or reply.identifier}')
        exp['report_uploaded_at'] = utc_now_iso()
        exp['report_upload_error'] = None
        summary = self._metrics_summary(performance) if remote_status == 'completed' else ''
        body_lines = [
            f'name={exp["name"]}',
            f'phase={exp.get("phase")}',
            f'run_dir={exp.get("run_dir")}',
        ]
        if summary:
            body_lines.append(f'metrics={summary}')
        if error_text:
            body_lines.append(f'error={error_text[:200]}')
        self._notify(
            exp,
            'uploaded' if remote_status == 'completed' else 'failed-uploaded',
            'Experiment Uploaded' if remote_status == 'completed' else 'Failed Experiment Uploaded',
            '\n'.join(body_lines),
        )

    def _sync_terminal_experiments(self):
        updated = False
        for exp in self.experiments:
            if not self._needs_remote_sync(exp):
                continue
            try:
                self._sync_terminal_experiment(exp)
            except Exception as exc:
                exp['report_upload_error'] = repr(exc)
                self._notify(
                    exp,
                    f'upload-failed:{repr(exc)[:80]}',
                    'Remote Upload Failed',
                    '\n'.join(
                        [
                            f'name={exp["name"]}',
                            f'phase={exp.get("phase")}',
                            f'status={exp.get("status")}',
                            f'error={repr(exc)}',
                        ]
                    ),
                )
                updated = True
                print(f'warning: failed to backfill remote upload for {exp["name"]}: {repr(exc)}')
            else:
                updated = True
        if updated:
            self._save_state()

    def _launch(self, exp: dict, gpu: dict):
        logical_args = build_args_for_phase(
            exp['base_args'],
            batch_size=int(exp['batch_size']),
            effective_batch_size=self.effective_batch_size,
            phase=str(exp['phase']),
            load_ckpt=exp.get('ckpt_path'),
        )
        args = deepcopy(logical_args)
        args['device'] = f'cuda:{gpu["index"]}'
        session = self._ensure_remote_session(exp)
        run_dir = run_dir_for_args(logical_args)
        log_name = sanitize_name(f"{exp['name']}__{exp['phase']}__b{exp['batch_size']}__r{exp['retries']}")
        log_path = self.logs_dir / f'{log_name}.log'
        command = trainer_command_from_args(args)
        env = os.environ.copy()
        env.setdefault('PYTHONUNBUFFERED', '1')
        if session:
            env['SECOMMENDER_REPORT_URI'] = self.server.uri
            env['SECOMMENDER_REPORT_AUTH_TOKEN'] = self.server.auth
            env['SECOMMENDER_REPORT_SESSION'] = session
            env['SECOMMENDER_REPORT_PHASE'] = str(exp['phase'])
            try:
                register_reply = self.server.register_experiment(
                    session,
                    run_dir=str(run_dir),
                    log_path=str(log_path),
                    command=' '.join(shlex.quote(part) for part in command),
                    phase=str(exp['phase']),
                )
                if not register_reply.ok:
                    print(
                        'warning: failed to pre-register remote experiment '
                        f'for {exp["name"]}: {register_reply.msg or register_reply.identifier}'
                    )
            except Exception as exc:
                print(f'warning: failed to pre-register remote experiment for {exp["name"]}: {repr(exc)}')
        with log_path.open('w') as handle:
            handle.write(f'# started_at={utc_now_iso()}\n')
            handle.write(f'# gpu={gpu["index"]} free_mb={gpu["free_mb"]} total_mb={gpu["total_mb"]}\n')
            handle.write(f'# command={" ".join(shlex.quote(part) for part in command)}\n\n')
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        exp['status'] = 'running'
        exp['run_dir'] = str(run_dir)
        exp['log_path'] = str(log_path)
        exp['started_at'] = utc_now_iso()
        exp['last_error'] = None
        exp['report_uploaded_at'] = None
        exp['report_upload_error'] = None
        self.active_jobs[process.pid] = {
            'process': process,
            'gpu': gpu['index'],
            'experiment': exp,
        }
        self._save_state()

    def _ensure_remote_session(self, exp: dict):
        if self.server is None or exp['phase'] == 'precheck':
            return None
        if exp.get('report_session'):
            return exp['report_session']

        evaluation_reply = self.server.create_or_get_evaluation(
            signature=exp['report_signature'],
            command=exp['report_command'],
            configuration=exp['report_configuration'],
            name=exp['name'],
        )
        if not evaluation_reply.ok:
            raise ValueError(f'failed to create evaluation: {evaluation_reply.msg or evaluation_reply.identifier}')
        experiment_reply = self.server.create_or_get_experiment(
            signature=exp['report_signature'],
            seed=int(exp['report_seed']),
        )
        if not experiment_reply.ok:
            raise ValueError(f'failed to create experiment: {experiment_reply.msg or experiment_reply.identifier}')
        exp['report_session'] = str(experiment_reply.body)
        self._save_state()
        return exp['report_session']

    def _mark_failed(self, exp: dict, error: str):
        exp['status'] = 'failed'
        exp['last_error'] = error
        exp['finished_at'] = utc_now_iso()
        exp['report_uploaded_at'] = None
        exp['report_upload_error'] = None
        self._notify(
            exp,
            f'failed:{exp.get("phase")}:{error[:80]}',
            'Experiment Failed',
            '\n'.join(
                [
                    f'name={exp["name"]}',
                    f'phase={exp.get("phase")}',
                    f'batch_size={exp.get("batch_size")}',
                    f'error={error}',
                ]
            ),
        )

    def _handle_success(self, exp: dict):
        if exp['phase'] == 'precheck':
            exp['phase'] = 'train'
            exp['status'] = 'pending'
            exp['finished_at'] = utc_now_iso()
            return
        exp['status'] = 'done'
        exp['finished_at'] = utc_now_iso()
        exp['report_uploaded_at'] = None

    def _handle_oom(self, exp: dict, log_tail: str):
        current_batch_size = int(exp['batch_size'])
        smaller_batch = next_smaller_batch_size(current_batch_size)
        if exp['phase'] == 'test':
            if smaller_batch is None:
                self._mark_failed(exp, 'OOM during test-only with no smaller batch size available')
                return
            exp['batch_size'] = smaller_batch
            exp['test_retries'] = int(exp.get('test_retries', 0)) + 1
            exp['status'] = 'pending'
            exp['finished_at'] = utc_now_iso()
            self._notify(
                exp,
                f'oom-test-{current_batch_size}-to-{smaller_batch}',
                'OOM Test Retry',
                '\n'.join(
                    [
                        f'name={exp["name"]}',
                        f'phase=test',
                        f'batch_size={current_batch_size}->{smaller_batch}',
                        'action=retry test-only with smaller batch',
                    ]
                ),
            )
            return

        if exp['phase'] == 'train' and exp.get('run_dir'):
            ckpt_path = Path(exp['run_dir']) / 'best.pt'
            if ckpt_path.exists() and log_reached_test(log_tail):
                if smaller_batch is None:
                    self._mark_failed(exp, 'OOM during final test with no smaller batch size available')
                    return
                exp['phase'] = 'test'
                exp['batch_size'] = smaller_batch
                exp['ckpt_path'] = str(ckpt_path)
                exp['test_retries'] = int(exp.get('test_retries', 0)) + 1
                exp['status'] = 'pending'
                exp['finished_at'] = utc_now_iso()
                self._notify(
                    exp,
                    f'oom-train-to-test-{current_batch_size}-to-{smaller_batch}',
                    'OOM Final Test Fallback',
                    '\n'.join(
                        [
                            f'name={exp["name"]}',
                            f'phase=train->test',
                            f'batch_size={current_batch_size}->{smaller_batch}',
                            'action=resume from best checkpoint and rerun final test',
                        ]
                    ),
                )
                return

        if smaller_batch is None:
            self._mark_failed(exp, 'OOM with no smaller batch size available')
            return

        exp['batch_size'] = smaller_batch
        exp['phase'] = 'precheck'
        exp['status'] = 'pending'
        exp['retries'] = int(exp.get('retries', 0)) + 1
        exp['finished_at'] = utc_now_iso()
        self._notify(
            exp,
            f'oom-precheck-{current_batch_size}-to-{smaller_batch}',
            'OOM Batch Reduction',
            '\n'.join(
                [
                    f'name={exp["name"]}',
                    f'phase={exp.get("phase")}',
                    f'batch_size={current_batch_size}->{smaller_batch}',
                    'action=rerun precheck with smaller batch',
                ]
            ),
        )

    def _handle_failure(self, exp: dict, returncode: int, log_tail: str):
        if log_has_oom(log_tail):
            self._handle_oom(exp, log_tail)
            return
        preview = log_tail.strip().splitlines()[-1] if log_tail.strip() else f'process exited {returncode}'
        self._mark_failed(exp, preview[:500])

    def _poll_active_jobs(self):
        finished_pids = []
        for pid, record in list(self.active_jobs.items()):
            process = record['process']
            returncode = process.poll()
            if returncode is None:
                continue
            finished_pids.append(pid)
            exp = record['experiment']
            log_tail = read_log_tail(Path(exp['log_path'])) if exp.get('log_path') else ''
            if returncode == 0:
                self._handle_success(exp)
            else:
                self._handle_failure(exp, returncode, log_tail)
        for pid in finished_pids:
            self.active_jobs.pop(pid, None)
        if finished_pids:
            self._save_state()

    def _launch_pending_jobs(self):
        available_gpus = [gpu for gpu in query_gpus() if gpu['index'] not in {record['gpu'] for record in self.active_jobs.values()}]
        pending = self._pending_experiments()
        for exp in pending:
            eligible_index = next(
                (
                    index for index, gpu in enumerate(available_gpus)
                    if int(gpu['free_mb']) >= int(exp['required_free_memory_mb'])
                ),
                None,
            )
            if eligible_index is None:
                continue
            gpu = available_gpus.pop(eligible_index)
            try:
                self._launch(exp, gpu)
            except Exception as exc:
                self._mark_failed(exp, f'launch failed: {exc}')
                self._save_state()

    def run(self):
        print(f'scheduler plan={self.plan_name} experiments={len(self.experiments)} output={self.output_dir}')
        if self.server is None and self._has_unsynced_completed_experiments():
            print(
                'warning: terminal experiments still need remote sync, '
                'but backend reporting is unavailable for this plan'
            )
        self._sync_terminal_experiments()
        while not self._all_terminal():
            self._poll_active_jobs()
            self._sync_terminal_experiments()
            self._launch_pending_jobs()
            if self._all_terminal():
                break
            time.sleep(self.poll_interval)

        self._sync_terminal_experiments()
        self._save_state()
        done = sum(1 for exp in self.experiments if exp['status'] == 'done')
        failed = sum(1 for exp in self.experiments if exp['status'] == 'failed')
        print(f'scheduler finished done={done} failed={failed} state={self.state_path}')


def main():
    parser = argparse.ArgumentParser(description='Batch experiment scheduler for Secommenders.')
    parser.add_argument('--plan', required=True, help='Path to scheduler plan YAML file.')
    args = parser.parse_args()

    scheduler = Scheduler(Path(args.plan))
    scheduler.run()


if __name__ == '__main__':
    main()
