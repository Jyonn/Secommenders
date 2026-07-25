#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduler import (  # noqa: E402
    ARTIFACT_ROOT,
    BATCH_LADDER,
    build_args_for_phase,
    initial_batch_cap,
    parse_trainer_command,
    run_dir_for_args,
    sanitize_name,
    trainer_command_from_args,
)


def load_plan_experiments(plan_path: Path):
    plan = yaml.safe_load(plan_path.read_text()) or {}
    raw_experiments = plan.get('experiments') or []
    if not raw_experiments:
        raise ValueError(f'No experiments found in plan {plan_path}')

    experiments = []
    for index, raw_experiment in enumerate(raw_experiments, start=1):
        if 'args' in raw_experiment:
            base_args = deepcopy(raw_experiment['args'])
        elif 'command' in raw_experiment:
            base_args = parse_trainer_command(raw_experiment['command'])
        else:
            raise ValueError(f'Experiment #{index} must provide either "args" or "command"')
        experiments.append({
            'name': sanitize_name(raw_experiment.get('name') or f'exp{index:03d}'),
            'base_args': base_args,
            'batch_size_cap': int(
                raw_experiment.get('batch_size_cap')
                or initial_batch_cap(str(base_args.get('model')))
            ),
        })
    return plan, experiments


def load_scheduler_state(plan: dict, plan_path: Path):
    plan_name = sanitize_name(plan.get('name') or plan_path.stem)
    state_path = ARTIFACT_ROOT / plan_name / 'state.json'
    if not state_path.exists():
        return {}, state_path
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return {}, state_path
    by_name = {
        str(experiment.get('name')): experiment
        for experiment in state.get('experiments') or []
        if experiment.get('name')
    }
    return by_name, state_path


def choose_batch_size(experiment: dict, state_experiment: dict | None, effective_batch_size: int):
    persisted_batch_size = (state_experiment or {}).get('batch_size')
    if persisted_batch_size is not None:
        persisted_batch_size = int(persisted_batch_size)
        if persisted_batch_size > 0 and effective_batch_size % persisted_batch_size == 0:
            return persisted_batch_size

    batch_cap = min(int(experiment.get('batch_size_cap') or effective_batch_size), effective_batch_size)
    for candidate in BATCH_LADDER:
        if candidate <= batch_cap and effective_batch_size % candidate == 0:
            return candidate
    return 1


def find_checkpoint(experiment: dict, state_experiment: dict | None, effective_batch_size: int):
    state_checkpoint = (state_experiment or {}).get('ckpt_path')
    if state_checkpoint and Path(state_checkpoint).exists():
        return Path(state_checkpoint)
    state_run_dir = (state_experiment or {}).get('run_dir')
    if state_run_dir:
        checkpoint = Path(state_run_dir) / 'best.pt'
        if checkpoint.exists():
            return checkpoint

    batch_size = choose_batch_size(experiment, state_experiment, effective_batch_size)
    train_args = build_args_for_phase(
        experiment['base_args'],
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        phase='train',
    )
    checkpoint = run_dir_for_args(train_args) / 'best.pt'
    return checkpoint if checkpoint.exists() else None


def build_frequency_args(
    experiment: dict,
    state_experiment: dict | None,
    *,
    effective_batch_size: int,
    checkpoint: Path,
    boundaries: str,
):
    batch_size = choose_batch_size(experiment, state_experiment, effective_batch_size)
    args = build_args_for_phase(
        experiment['base_args'],
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        phase='test',
        load_ckpt=checkpoint,
    )
    args['frequency_breakdown'] = True
    args['frequency_buckets'] = boundaries
    args['overwrite'] = 'true'
    args['num_gpus'] = 1
    return args


def format_metric(value):
    return '-' if value is None else f'{float(value):.4f}'


def print_breakdown(name: str, analysis_path: Path):
    payload = json.loads(analysis_path.read_text())
    print(f'\n=== {name} ===', flush=True)
    print(
        f'data={payload.get("data")} model={payload.get("model")} '
        f'repr={payload.get("repr_type")} task={payload.get("task_type")} '
        f'test_targets={payload.get("total_test_targets")}',
        flush=True,
    )
    ks = sorted({
        int(key.split('@', 1)[1])
        for bucket in payload.get('buckets', {}).values()
        for key in bucket
        if key.startswith('hr@')
    })
    header = ['frequency', '#target', 'share', 'mrr']
    for k in ks:
        header.extend([f'hr@{k}', f'ndcg@{k}'])
    print('  '.join(f'{column:>12}' for column in header), flush=True)
    for label, bucket in payload.get('buckets', {}).items():
        share = bucket.get('target_share')
        values = [
            label,
            str(bucket.get('target_count', 0)),
            '-' if share is None else f'{float(share):.2%}',
            format_metric(bucket.get('mrr')),
        ]
        for k in ks:
            values.extend([
                format_metric(bucket.get(f'hr@{k}')),
                format_metric(bucket.get(f'ndcg@{k}')),
            ])
        print('  '.join(f'{value:>12}' for value in values), flush=True)
    print(f'analysis={analysis_path}', flush=True)


def write_manifest(path: Path, plan_path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'plan': str(plan_path),
        'experiments': entries,
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')


def has_per_target_records(path: Path):
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(payload.get('records'), list) and bool(payload['records'])


def main():
    parser = argparse.ArgumentParser(
        description='Compute target-frequency ranking breakdowns for all trained experiments in a scheduler plan.'
    )
    parser.add_argument('--plan', required=True, help='Path to scheduler plan YAML file.')
    parser.add_argument(
        '--frequency-buckets',
        default='0,5,20,100',
        help='Absolute finetune target-frequency boundaries (default: 0,5,20,100).',
    )
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan, experiments = load_plan_experiments(plan_path)
    effective_batch_size = int(plan.get('effective_batch_size', 64))
    state_by_name, state_path = load_scheduler_state(plan, plan_path)
    manifest_path = state_path.parent / 'frequency_breakdown_manifest.json'

    print(
        f'frequency breakdown plan={plan.get("name") or plan_path.stem} '
        f'experiments={len(experiments)} state={state_path}',
        flush=True,
    )

    completed = 0
    skipped = 0
    failed = 0
    manifest_entries = []
    write_manifest(manifest_path, plan_path, manifest_entries)
    for index, experiment in enumerate(experiments, start=1):
        name = experiment['name']
        state_experiment = state_by_name.get(name)
        checkpoint = find_checkpoint(experiment, state_experiment, effective_batch_size)
        print(f'\n[{index}/{len(experiments)}] {name}', flush=True)
        if checkpoint is None:
            skipped += 1
            print('skip: trained checkpoint best.pt not found', flush=True)
            continue

        frequency_args = build_frequency_args(
            experiment,
            state_experiment,
            effective_batch_size=effective_batch_size,
            checkpoint=checkpoint,
            boundaries=args.frequency_buckets,
        )
        test_run_dir = run_dir_for_args(frequency_args)
        analysis_path = test_run_dir / 'analysis' / 'frequency_breakdown_test.json'
        command = trainer_command_from_args(frequency_args)
        print(f'checkpoint={checkpoint}', flush=True)
        if has_per_target_records(analysis_path):
            print(f'reuse per-target analysis={analysis_path}', flush=True)
        else:
            print(f'command={" ".join(command)}', flush=True)
            env = os.environ.copy()
            env.setdefault('PYTHONUNBUFFERED', '1')
            for key in (
                'SECOMMENDER_REPORT_URI',
                'SECOMMENDER_REPORT_AUTH_TOKEN',
                'SECOMMENDER_REPORT_SESSION',
                'SECOMMENDER_REPORT_PHASE',
            ):
                env.pop(key, None)
            result = subprocess.run(command, cwd=ROOT, env=env)
            if result.returncode != 0:
                failed += 1
                print(f'failed: trainer exited with code {result.returncode}', flush=True)
                continue
        if not analysis_path.exists():
            failed += 1
            print(f'failed: analysis was not written to {analysis_path}', flush=True)
            continue
        print_breakdown(name, analysis_path)
        manifest_entries.append({
            'name': name,
            'args': experiment['base_args'],
            'analysis_path': str(analysis_path),
            'checkpoint': str(checkpoint),
        })
        write_manifest(manifest_path, plan_path, manifest_entries)
        completed += 1

    print(
        f'\nfrequency breakdown finished completed={completed} skipped={skipped} failed={failed}',
        flush=True,
    )
    print(f'manifest={manifest_path}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
