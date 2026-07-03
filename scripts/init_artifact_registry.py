import argparse
import json
import shutil
import sys
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
            if apply:
                try:
                    final_meta_path = meta_path
                    if meta_path.parent != target_run_dir:
                        if target_run_dir.exists():
                            raise ValueError(f'target run dir already exists: {target_run_dir}')
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
                    register_trained_artifact(config, target_run_dir, aliases=identity.get('aliases'), root=root)
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
                        {
                            'data': dataset,
                            'folder': folder,
                            'signature': signature,
                            'error': str(exc),
                        }
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
    parser.add_argument('--json', action='store_true', help='Print the full report as JSON.')
    args = parser.parse_args()

    report = init_trained_registry(
        Path(args.root),
        data=args.data,
        apply=bool(args.apply),
        delete_abnormal_empty=bool(args.delete_abnormal_empty),
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
        print(f'  conflict {item["data"]}/{item["folder"]}: {item["error"]}')
    for item in report['moved'][:20]:
        print(f'  moved {item["from"]} -> {item["to"]}')
    for item in report['delete_candidates'][:20]:
        reasons = ','.join(item['reasons'])
        print(f'  delete-candidate {item["data"]}/{item["folder"]}: {reasons}')


if __name__ == '__main__':
    main()
