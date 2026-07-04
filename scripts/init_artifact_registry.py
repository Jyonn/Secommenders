import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.artifact_identity import (  # noqa: E402
    TRAINED_INDEX_NAME,
    canonical_trained_run_dir,
    legacy_signature_from_folder,
    load_trained_index,
    migrate_train_config_dict,
    register_trained_artifact,
    save_trained_index,
    trained_artifact_identity,
    trained_mode,
    trained_seed,
    trained_signature_from_config,
)


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid json: {exc}') from exc


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def iter_trained_meta_paths(root: Path, data: str | None = None):
    base = root / 'artifacts' / 'trained'
    if data:
        dataset_dir = base / data.lower()
        paths = sorted(dataset_dir.glob('*/meta.json')) + sorted(dataset_dir.glob('*/*/meta.json'))
        yield from paths
        return
    paths = sorted(base.glob('*/*/meta.json')) + sorted(base.glob('*/*/*/meta.json'))
    yield from paths


def is_nested_trained_meta(root: Path, meta_path: Path):
    try:
        relative = meta_path.relative_to(root / 'artifacts' / 'trained')
    except ValueError:
        return False
    return len(relative.parts) == 4


def collect_existing_identity_aliases(meta: dict):
    identity = meta.get('artifact_identity')
    if not isinstance(identity, dict):
        return []
    aliases = []
    raw_aliases = identity.get('aliases') or []
    if isinstance(raw_aliases, list):
        aliases.extend(str(alias) for alias in raw_aliases)
    return aliases


def update_trained_meta(meta_path: Path, config, *, apply: bool, aliases: list[str] | None = None):
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError('meta root must be a dict')
    if not isinstance(meta.get('config'), dict):
        raise ValueError('missing meta.config')

    run_dir = meta_path.parent
    aliases = [*(aliases or []), *collect_existing_identity_aliases(meta)]
    legacy_signature = legacy_signature_from_folder(run_dir.name)
    if legacy_signature:
        aliases.append(legacy_signature)
    identity = trained_artifact_identity(
        config,
        run_dir,
        aliases=aliases,
        migration_status='migrated',
    )
    if apply:
        meta['artifact_identity'] = identity
        write_json(meta_path, meta)
    return identity


def checkpoint_exists(setting_dir: Path):
    if not setting_dir.exists():
        return False
    return any(path.name == 'best.pt' for path in setting_dir.rglob('best.pt'))


def summarize_metrics(metrics: dict | None):
    if not isinstance(metrics, dict):
        return {}
    summary = {}
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float, str)):
            summary[key] = value
    return summary


def summarize_run_dir(run_dir: Path):
    run_dir = Path(run_dir)
    meta_path = run_dir / 'meta.json'
    pid_path = run_dir / 'pid.json'
    checkpoint_path = run_dir / 'best.pt'
    summary = {
        'path': str(run_dir),
        'exists': run_dir.exists(),
        'has_meta': meta_path.exists(),
        'has_checkpoint': checkpoint_path.exists(),
        'has_pid': pid_path.exists(),
    }
    if checkpoint_path.exists():
        summary['checkpoint_size_bytes'] = checkpoint_path.stat().st_size
        summary['checkpoint_mtime'] = datetime.fromtimestamp(
            checkpoint_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
    if not meta_path.exists():
        return summary
    try:
        meta = read_json(meta_path)
    except ValueError as exc:
        summary['meta_error'] = str(exc)
        return summary
    if not isinstance(meta, dict):
        summary['meta_error'] = 'meta root is not a dict'
        return summary
    identity = meta.get('artifact_identity') if isinstance(meta.get('artifact_identity'), dict) else {}
    config = meta.get('config') if isinstance(meta.get('config'), dict) else {}
    summary.update(
        {
            'status': meta.get('status'),
            'mode': identity.get('mode'),
            'seed': identity.get('seed') if identity.get('seed') is not None else config.get('seed'),
            'signature': identity.get('signature'),
            'schema_version': identity.get('schema_version'),
            'aliases': identity.get('aliases') or [],
            'best_epoch': meta.get('best_epoch'),
            'main_metric': meta.get('main_metric'),
            'best_valid_metric': meta.get('best_valid_metric'),
            'finished_at': meta.get('finished_at'),
            'started_at': meta.get('started_at'),
            'error': meta.get('error'),
            'failed_at': meta.get('failed_at'),
            'test_metrics': summarize_metrics(meta.get('test_metrics')),
            'valid_metrics': summarize_metrics(meta.get('valid_metrics')),
        }
    )
    if config:
        summary['config_brief'] = {
            key: config.get(key)
            for key in (
                'data',
                'model',
                'repr_type',
                'task_type',
                'seed',
                'batch_size',
                'accumulate_batch',
                'learning_rate',
                'weight_decay',
            )
            if key in config
        }
    return summary


def compact_json(value):
    if value in (None, {}, []):
        return '-'
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def print_run_summary(label: str, summary: dict):
    print(
        f'    {label}: status={summary.get("status") or "-"} mode={summary.get("mode") or "-"} '
        f'seed={summary.get("seed") if summary.get("seed") is not None else "-"} '
        f'ckpt={summary.get("has_checkpoint")} pid={summary.get("has_pid")} '
        f'best={summary.get("best_valid_metric") if summary.get("best_valid_metric") is not None else "-"} '
        f'test={compact_json(summary.get("test_metrics"))}'
    )
    print(f'      path={summary.get("path")}')
    if summary.get('checkpoint_mtime'):
        print(
            f'      checkpoint size={summary.get("checkpoint_size_bytes")} '
            f'mtime={summary.get("checkpoint_mtime")}'
        )
    if summary.get('config_brief'):
        print(f'      config={compact_json(summary.get("config_brief"))}')
    if summary.get('error') or summary.get('failed_at') or summary.get('meta_error'):
        print(
            f'      error={summary.get("error") or "-"} failed_at={summary.get("failed_at") or "-"} '
            f'meta_error={summary.get("meta_error") or "-"}'
        )


def conflict_report(data: str, folder: str, signature: str, source_dir: Path, target_dir: Path, *, error: str):
    return {
        'data': data,
        'folder': folder,
        'signature': signature,
        'error': error,
        'source': summarize_run_dir(source_dir),
        'target': summarize_run_dir(target_dir),
    }


def backup_existing_target(target_run_dir: Path):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = target_run_dir.with_name(f'{target_run_dir.name}.conflict-{timestamp}')
    suffix = 1
    while backup.exists():
        backup = target_run_dir.with_name(f'{target_run_dir.name}.conflict-{timestamp}-{suffix}')
        suffix += 1
    shutil.move(str(target_run_dir), str(backup))
    return backup


def collect_delete_candidates(root: Path, data: str | None = None):
    base = root / 'artifacts' / 'trained'
    dataset_dirs = [base / data.lower()] if data else sorted(path for path in base.glob('*') if path.is_dir())
    candidates = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        for setting_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            if checkpoint_exists(setting_dir):
                continue
            meta_paths = sorted(setting_dir.glob('*/meta.json')) + sorted([setting_dir / 'meta.json'])
            train_like = False
            abnormal_reasons = []
            for meta_path in meta_paths:
                if not meta_path.exists():
                    continue
                try:
                    meta = read_json(meta_path)
                except ValueError as exc:
                    abnormal_reasons.append(str(exc))
                    continue
                identity = meta.get('artifact_identity') if isinstance(meta, dict) else None
                mode = identity.get('mode') if isinstance(identity, dict) else None
                if mode is None and isinstance(meta.get('config'), dict):
                    try:
                        mode = trained_mode(migrate_train_config_dict(meta['config']))
                    except ValueError:
                        mode = None
                train_like = train_like or mode == 'train'
                status = str(meta.get('status') or '').lower() if isinstance(meta, dict) else ''
                if status in {'failed', 'running'} or meta.get('error') or meta.get('failed_at'):
                    abnormal_reasons.append(f'status={status or "unknown"}')
            if train_like and abnormal_reasons:
                candidates.append(
                    {
                        'data': dataset_dir.name,
                        'folder': setting_dir.name,
                        'path': str(setting_dir),
                        'reasons': sorted(set(abnormal_reasons)),
                    }
                )
    return candidates


def init_trained_registry(
    root: Path,
    *,
    data: str | None,
    apply: bool,
    delete_abnormal_empty: bool = False,
    resolve_conflict: str = 'report',
):
    resolved = []
    unresolved = []
    conflicts = []
    moved = []

    for meta_path in iter_trained_meta_paths(root, data=data):
        if not meta_path.exists():
            continue
        dataset = meta_path.parent.parent.name
        if is_nested_trained_meta(root, meta_path):
            dataset = meta_path.parent.parent.parent.name
        folder = meta_path.parent.name
        try:
            meta = read_json(meta_path)
            config = migrate_train_config_dict(meta.get('config'))
            signature = trained_signature_from_config(config)
            aliases = []
            if not is_nested_trained_meta(root, meta_path):
                legacy_signature = legacy_signature_from_folder(meta_path.parent.name)
                if legacy_signature:
                    aliases.append(legacy_signature)
            identity = update_trained_meta(meta_path, config, apply=False, aliases=aliases)
            target_run_dir = canonical_trained_run_dir(config, root=root)
            resolved.append(
                {
                    'data': dataset,
                    'folder': folder,
                    'signature': signature,
                    'seed': trained_seed(config),
                    'target_run_dir': str(target_run_dir),
                    'aliases': identity.get('aliases') or [],
                    'meta_path': str(meta_path),
                }
            )
            if not apply and meta_path.parent != target_run_dir and target_run_dir.exists():
                conflict = {
                    'data': dataset,
                    'folder': folder,
                    'signature': signature,
                    'seed': trained_seed(config),
                    'source': summarize_run_dir(meta_path.parent),
                    'target': summarize_run_dir(target_run_dir),
                    'resolution': 'dry-run',
                    'error': f'target run dir already exists: {target_run_dir}',
                }
                conflicts.append(conflict)
            if apply:
                try:
                    final_meta_path = meta_path
                    if meta_path.parent != target_run_dir:
                        existing_target_conflict = None
                        if target_run_dir.exists():
                            existing_target_conflict = {
                                'data': dataset,
                                'folder': folder,
                                'signature': signature,
                                'seed': trained_seed(config),
                                'source': summarize_run_dir(meta_path.parent),
                                'target': summarize_run_dir(target_run_dir),
                                'resolution': resolve_conflict,
                            }
                            if resolve_conflict == 'report':
                                existing_target_conflict['error'] = f'target run dir already exists: {target_run_dir}'
                                conflicts.append(existing_target_conflict)
                                continue
                            if resolve_conflict == 'keep-existing':
                                register_trained_artifact(
                                    config,
                                    target_run_dir,
                                    aliases=identity.get('aliases'),
                                    root=root,
                                )
                                existing_target_conflict['action'] = 'kept existing target and skipped source'
                                conflicts.append(existing_target_conflict)
                                continue
                            if resolve_conflict == 'keep-source':
                                backup = backup_existing_target(target_run_dir)
                                existing_target_conflict['action'] = 'backed up existing target and moved source'
                                existing_target_conflict['target_backup'] = str(backup)
                                conflicts.append(existing_target_conflict)
                            else:
                                raise ValueError(f'unsupported conflict resolution: {resolve_conflict}')
                        old_run_dir = meta_path.parent
                        old_setting_dir = old_run_dir.parent
                        target_run_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(old_run_dir), str(target_run_dir))
                        if old_setting_dir.exists() and not any(old_setting_dir.iterdir()):
                            old_setting_dir.rmdir()
                        final_meta_path = target_run_dir / 'meta.json'
                        moved.append(
                            {
                                'data': dataset,
                                'from': str(old_run_dir),
                                'to': str(target_run_dir),
                            }
                        )
                    try:
                        register_trained_artifact(config, target_run_dir, aliases=identity.get('aliases'), root=root)
                    except ValueError as exc:
                        conflicts.append(
                            conflict_report(
                                dataset,
                                folder,
                                signature,
                                meta_path.parent,
                                target_run_dir,
                                error=str(exc),
                            )
                        )
                        continue
                    final_meta = read_json(final_meta_path)
                    final_meta['run_dir'] = str(target_run_dir)
                    final_meta['artifact_identity'] = trained_artifact_identity(
                        config,
                        target_run_dir,
                        aliases=identity.get('aliases'),
                        migration_status='migrated',
                    )
                    write_json(final_meta_path, final_meta)
                except ValueError as exc:
                    conflicts.append(
                        conflict_report(
                            dataset,
                            folder,
                            signature,
                            meta_path.parent,
                            target_run_dir,
                            error=str(exc),
                        )
                    )
        except Exception as exc:
            unresolved.append(
                {
                    'data': dataset,
                    'folder': folder,
                    'meta_path': str(meta_path),
                    'reason': str(exc),
                }
            )

    touched_data = sorted({item['data'] for item in resolved})
    if apply:
        for dataset in touched_data:
            index = load_trained_index(dataset, root=root)
            save_trained_index(dataset, index, root=root)

    delete_candidates = collect_delete_candidates(root, data=data)
    deleted = []
    if apply and delete_abnormal_empty:
        for candidate in delete_candidates:
            path = Path(candidate['path'])
            if path.exists():
                shutil.rmtree(path)
                deleted.append(candidate)

    return {
        'stage': 'trained',
        'mode': 'apply' if apply else 'dry-run',
        'resolved_count': len(resolved),
        'unresolved_count': len(unresolved),
        'conflict_count': len(conflicts),
        'moved_count': len(moved),
        'delete_candidate_count': len(delete_candidates),
        'deleted_count': len(deleted),
        'index_name': TRAINED_INDEX_NAME,
        'datasets': touched_data,
        'resolved': resolved,
        'unresolved': unresolved,
        'conflicts': conflicts,
        'moved': moved,
        'delete_candidates': delete_candidates,
        'deleted': deleted,
    }


def main():
    parser = argparse.ArgumentParser(description='Initialize artifact registry indexes for existing artifacts.')
    parser.add_argument('--stage', default='trained', choices=['trained'])
    parser.add_argument('--data', default=None, help='Optional dataset name, e.g. mind.')
    parser.add_argument('--root', default=str(ROOT), help='Algorithm project root. Defaults to this repository.')
    parser.add_argument('--apply', action='store_true', help='Write meta.json artifact_identity fields and .index.json.')
    parser.add_argument(
        '--delete-abnormal-empty',
        action='store_true',
        help='With --apply, delete train-mode setting folders that have no best.pt and look failed/running.',
    )
    parser.add_argument(
        '--resolve-conflict',
        choices=['report', 'keep-existing', 'keep-source'],
        default='report',
        help='How to handle migration conflicts when the canonical seed directory already exists.',
    )
    parser.add_argument('--json', action='store_true', help='Print the full report as JSON.')
    args = parser.parse_args()

    report = init_trained_registry(
        Path(args.root),
        data=args.data,
        apply=bool(args.apply),
        delete_abnormal_empty=bool(args.delete_abnormal_empty),
        resolve_conflict=str(args.resolve_conflict),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(
        f'artifact registry init stage={report["stage"]} mode={report["mode"]} '
        f'resolved={report["resolved_count"]} unresolved={report["unresolved_count"]} '
        f'conflicts={report["conflict_count"]} moved={report["moved_count"]} '
        f'delete_candidates={report["delete_candidate_count"]} deleted={report["deleted_count"]}'
    )
    if report['datasets']:
        print('datasets: ' + ', '.join(report['datasets']))
    for item in report['resolved'][:20]:
        aliases = ','.join(item['aliases'])
        print(f'  ok {item["data"]}/{item["folder"]} -> {item["signature"]} aliases=[{aliases}]')
    if len(report['resolved']) > 20:
        print(f'  ... {len(report["resolved"]) - 20} more resolved')
    for item in report['unresolved'][:20]:
        print(f'  unresolved {item["data"]}/{item["folder"]}: {item["reason"]}')
    if len(report['unresolved']) > 20:
        print(f'  ... {len(report["unresolved"]) - 20} more unresolved')
    for item in report['conflicts'][:20]:
        print(f'  conflict {item["data"]}/{item["folder"]}: {item.get("error") or item.get("action")}')
        source = item.get('source') or {}
        target = item.get('target') or {}
        if source or target:
            print_run_summary('source', source)
            print_run_summary('target', target)
            print(
                '    choose: --resolve-conflict keep-existing to keep target, '
                'or --resolve-conflict keep-source to back up target and move source'
            )
    for item in report['moved'][:20]:
        print(f'  moved {item["from"]} -> {item["to"]}')
    deleted_paths = {item.get('path') for item in report.get('deleted', [])}
    for item in report['delete_candidates'][:20]:
        reasons = ','.join(item['reasons'])
        marker = 'deleted' if item.get('path') in deleted_paths else 'not deleted'
        print(f'  delete-candidate ({marker}) {item["data"]}/{item["folder"]}: {reasons}')
    if report['delete_candidate_count'] and not report['deleted_count']:
        print('  note: delete candidates are only reported by default; add --apply --delete-abnormal-empty to delete them.')


if __name__ == '__main__':
    main()
