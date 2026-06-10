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

from core.train_config import TrainConfig
from utils.config_init import ConfigInit
from utils.model import match as match_model_key


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / 'artifacts' / 'scheduler'
BATCH_LADDER = [64, 32, 16, 8, 4, 2, 1]
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


def classify_model_size(model_name: str):
    model_name = str(model_name).lower()
    mapped = (match_model_key(model_name) or '').lower()
    combined = f'{model_name} {mapped}'
    if model_name == 'scratch':
        return 'scratch'
    if any(token in combined for token in ['0.8b', '08b']):
        return '0.8b'
    if any(token in combined for token in ['7b', '8b', '9b']):
        return '8b'
    if any(token in combined for token in ['4b', '3b', '2b', '1.3b', '1b']):
        return '4b'
    return '4b'


def initial_batch_cap(model_name: str):
    bucket = classify_model_size(model_name)
    if bucket == 'scratch':
        return 64
    if bucket == '0.8b':
        return 32
    if bucket == '4b':
        return 16
    return 4


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


class Scheduler:
    def __init__(self, plan_path: Path):
        self.plan_path = Path(plan_path)
        self.plan = json.loads(self.plan_path.read_text())
        self.plan_name = sanitize_name(self.plan.get('name') or self.plan_path.stem)
        self.poll_interval = int(self.plan.get('poll_interval_seconds', 15))
        self.effective_batch_size = int(self.plan.get('effective_batch_size', 64))
        self.output_dir = ARTIFACT_ROOT / self.plan_name
        self.logs_dir = self.output_dir / 'logs'
        self.state_path = self.output_dir / 'state.json'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = self._load_or_initialize_state()
        self.active_jobs = {}

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
        return {
            'name': name,
            'priority': int(raw_exp.get('priority', 0)),
            'base_args': base_args,
            'batch_size_cap': batch_cap,
            'batch_size': batch_size,
            'phase': 'precheck',
            'status': 'pending',
            'retries': 0,
            'test_retries': 0,
            'run_dir': None,
            'ckpt_path': None,
            'log_path': None,
            'last_error': None,
            'started_at': None,
            'finished_at': None,
        }

    def _load_or_initialize_state(self):
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            experiments = state.get('experiments', [])
            for exp in experiments:
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

    def _launch(self, exp: dict, gpu: dict):
        args = build_args_for_phase(
            exp['base_args'],
            batch_size=int(exp['batch_size']),
            effective_batch_size=self.effective_batch_size,
            phase=str(exp['phase']),
            load_ckpt=exp.get('ckpt_path'),
        )
        run_dir = run_dir_for_args(args)
        log_name = sanitize_name(f"{exp['name']}__{exp['phase']}__b{exp['batch_size']}__r{exp['retries']}")
        log_path = self.logs_dir / f'{log_name}.log'
        command = trainer_command_from_args(args)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu['index'])
        env.setdefault('PYTHONUNBUFFERED', '1')
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
        self.active_jobs[process.pid] = {
            'process': process,
            'gpu': gpu['index'],
            'experiment': exp,
        }
        self._save_state()

    def _mark_failed(self, exp: dict, error: str):
        exp['status'] = 'failed'
        exp['last_error'] = error
        exp['finished_at'] = utc_now_iso()

    def _handle_success(self, exp: dict):
        if exp['phase'] == 'precheck':
            exp['phase'] = 'train'
            exp['status'] = 'pending'
            exp['finished_at'] = utc_now_iso()
            return
        exp['status'] = 'done'
        exp['finished_at'] = utc_now_iso()

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
                return

        if smaller_batch is None:
            self._mark_failed(exp, 'OOM with no smaller batch size available')
            return

        exp['batch_size'] = smaller_batch
        exp['phase'] = 'precheck'
        exp['status'] = 'pending'
        exp['retries'] = int(exp.get('retries', 0)) + 1
        exp['finished_at'] = utc_now_iso()

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
        for gpu, exp in zip(available_gpus, pending):
            self._launch(exp, gpu)

    def run(self):
        print(f'scheduler plan={self.plan_name} experiments={len(self.experiments)} output={self.output_dir}')
        while not self._all_terminal():
            self._poll_active_jobs()
            self._launch_pending_jobs()
            if self._all_terminal():
                break
            time.sleep(self.poll_interval)

        self._save_state()
        done = sum(1 for exp in self.experiments if exp['status'] == 'done')
        failed = sum(1 for exp in self.experiments if exp['status'] == 'failed')
        print(f'scheduler finished done={done} failed={failed} state={self.state_path}')


def main():
    parser = argparse.ArgumentParser(description='Batch experiment scheduler for Secommenders.')
    parser.add_argument('--plan', required=True, help='Path to scheduler plan JSON file.')
    args = parser.parse_args()

    scheduler = Scheduler(Path(args.plan))
    scheduler.run()


if __name__ == '__main__':
    main()
