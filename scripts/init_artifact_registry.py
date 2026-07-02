import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.artifact_identity import (  # noqa: E402
    TRAINED_INDEX_NAME,
    legacy_signature_from_folder,
    load_trained_index,
    migrate_train_config_dict,
    register_trained_artifact,
    save_trained_index,
    trained_artifact_identity,
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
        yield from sorted((base / data.lower()).glob('*/meta.json'))
        return
    yield from sorted(base.glob('*/*/meta.json'))


def collect_existing_identity_aliases(meta: dict):
    identity = meta.get('artifact_identity')
    if not isinstance(identity, dict):
        return []
    aliases = []
    if identity.get('signature'):
        aliases.append(str(identity['signature']))
    raw_aliases = identity.get('aliases') or []
    if isinstance(raw_aliases, list):
        aliases.extend(str(alias) for alias in raw_aliases)
    return aliases


def update_trained_meta(meta_path: Path, config, *, apply: bool):
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError('meta root must be a dict')
    if not isinstance(meta.get('config'), dict):
        raise ValueError('missing meta.config')

    run_dir = meta_path.parent
    aliases = collect_existing_identity_aliases(meta)
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


def init_trained_registry(root: Path, *, data: str | None, apply: bool):
    resolved = []
    unresolved = []
    conflicts = []

    for meta_path in iter_trained_meta_paths(root, data=data):
        dataset = meta_path.parent.parent.name
        folder = meta_path.parent.name
        try:
            meta = read_json(meta_path)
            config = migrate_train_config_dict(meta.get('config'))
            signature = trained_signature_from_config(config)
            identity = update_trained_meta(meta_path, config, apply=False)
            resolved.append(
                {
                    'data': dataset,
                    'folder': folder,
                    'signature': signature,
                    'aliases': identity.get('aliases') or [],
                    'meta_path': str(meta_path),
                }
            )
            if apply:
                try:
                    register_trained_artifact(config, meta_path.parent, aliases=identity.get('aliases'), root=root)
                    update_trained_meta(meta_path, config, apply=True)
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

    return {
        'stage': 'trained',
        'mode': 'apply' if apply else 'dry-run',
        'resolved_count': len(resolved),
        'unresolved_count': len(unresolved),
        'conflict_count': len(conflicts),
        'index_name': TRAINED_INDEX_NAME,
        'datasets': touched_data,
        'resolved': resolved,
        'unresolved': unresolved,
        'conflicts': conflicts,
    }


def main():
    parser = argparse.ArgumentParser(description='Initialize artifact registry indexes for existing artifacts.')
    parser.add_argument('--stage', default='trained', choices=['trained'])
    parser.add_argument('--data', default=None, help='Optional dataset name, e.g. mind.')
    parser.add_argument('--root', default=str(ROOT), help='Algorithm project root. Defaults to this repository.')
    parser.add_argument('--apply', action='store_true', help='Write meta.json artifact_identity fields and .index.json.')
    parser.add_argument('--json', action='store_true', help='Print the full report as JSON.')
    args = parser.parse_args()

    report = init_trained_registry(Path(args.root), data=args.data, apply=bool(args.apply))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(
        f'artifact registry init stage={report["stage"]} mode={report["mode"]} '
        f'resolved={report["resolved_count"]} unresolved={report["unresolved_count"]} '
        f'conflicts={report["conflict_count"]}'
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


if __name__ == '__main__':
    main()
