import argparse
import json
from pathlib import Path

from utils.artifact_identity import normalize_legacy_train_config


ROOT = Path(__file__).resolve().parent

RUNTIME_KEYS = {'root', 'json', 'all_seeds', 'limit', 'overwrite', 'load_ckpt', 'device'}
FILTER_KEYS = [
    'data',
    'model',
    'repr_type',
    'task_type',
    'repr_source_model',
    'sid_export',
    'sid_coder',
    'hash_coder',
    'repr_combine',
    'maxitems',
    'model_max_length',
    'item_text_max_tokens',
    'batch_size',
    'accumulate_batch',
    'learning_rate',
    'weight_decay',
    'seed',
    'uid_decoding',
    'uid_cluster_levels',
    'uid_cluster_topk',
    'code_decoding',
    'code_beam_width',
    'code_beam_chunk_size',
    'code_collision_loss_weight',
    'main_metric',
    'metrics',
    'patience',
    'alignment_weight',
    'num_gpus',
    'freeze_backbone',
    'model_dtype',
    'use_lora',
    'lora_rank',
    'lora_alpha',
    'lora_dropout',
    'lora_layers',
    'lora_target_modules',
    'hidden_size',
    'num_layers',
    'num_heads',
    'dropout',
    'epochs',
    'valid_only',
    'test_only',
]


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def coerce_scalar(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    lower = text.lower()
    if lower == 'null':
        return None
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    if text.isdigit() or (text.startswith('-') and text[1:].isdigit()):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return value


def normalize_aliases(kwargs: dict):
    aliases = {
        'repr': 'repr_type',
        'task': 'task_type',
        'lr': 'learning_rate',
        'wd': 'weight_decay',
        'sid_quantizer_name': 'sid_coder',
        'hash_quantizer_name': 'hash_coder',
    }
    normalized = dict(kwargs)
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    return normalized


def cli_filters(args):
    raw = {
        key: coerce_scalar(value)
        for key, value in vars(args).items()
        if key not in RUNTIME_KEYS and value is not None
    }
    raw = normalize_aliases(raw)
    return {key: value for key, value in raw.items() if key in FILTER_KEYS}


def normalize_for_compare(value):
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered == 'null':
            return None
        if ',' in text:
            return [normalize_for_compare(part) for part in text.split(',') if part.strip()]
        return lowered
    if isinstance(value, list):
        return [normalize_for_compare(item) for item in value]
    return value


def values_match(actual, expected):
    actual = normalize_for_compare(actual)
    expected = normalize_for_compare(expected)
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def iter_meta_paths(root: Path, data: str | None = None):
    base = root / 'artifacts' / 'trained'
    if data:
        dataset_dirs = [base / str(data).lower()]
    else:
        dataset_dirs = sorted(path for path in base.glob('*') if path.is_dir()) if base.exists() else []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        yield from sorted(dataset_dir.glob('*/*/*/meta.json'))
        yield from sorted(dataset_dir.glob('*/*/meta.json'))
        yield from sorted(dataset_dir.glob('*/meta.json'))


def trained_relative_parts(meta_path: Path):
    parts = meta_path.parts
    if 'trained' not in parts:
        return []
    trained_index = len(parts) - 1 - parts[::-1].index('trained')
    return list(parts[trained_index + 1 : -1])


def run_phase_from_path(meta_path: Path):
    parts = trained_relative_parts(meta_path)
    if len(parts) >= 4 and parts[3] in {'train', 'precheck', 'test'}:
        return parts[3]
    if len(parts) >= 3 and parts[2] in {'train', 'precheck', 'test'}:
        return parts[2]
    return '-'


def run_seed_from_path(meta_path: Path):
    parts = trained_relative_parts(meta_path)
    if len(parts) >= 4:
        return parts[2]
    return '-'


def path_value(meta_path: Path, offset: int, default='-'):
    parts = trained_relative_parts(meta_path)
    if 0 <= offset < len(parts):
        return parts[offset]
    return default


def row_from_meta(meta_path: Path, root: Path):
    meta = read_json_if_exists(meta_path) or {}
    raw_config = meta.get('config') if isinstance(meta.get('config'), dict) else {}
    config = normalize_legacy_train_config(raw_config) if raw_config else {}
    identity = meta.get('artifact_identity') if isinstance(meta.get('artifact_identity'), dict) else {}
    metrics = meta.get('test_metrics') if isinstance(meta.get('test_metrics'), dict) else None
    if metrics is None:
        metrics = meta.get('valid_metrics') if isinstance(meta.get('valid_metrics'), dict) else {}
    run_dir = meta_path.parent
    signature = identity.get('signature') or path_value(meta_path, 1)
    folder = identity.get('folder') or path_value(meta_path, 1)
    try:
        display_path = str(run_dir.relative_to(root))
    except ValueError:
        display_path = str(run_dir)
    return {
        'data': config.get('data') or identity.get('spec', {}).get('config', {}).get('data') or path_value(meta_path, 0),
        'model': config.get('model'),
        'repr_type': config.get('repr_type'),
        'task_type': config.get('task_type'),
        'seed': config.get('seed') if config.get('seed') is not None else run_seed_from_path(meta_path),
        'phase': identity.get('phase') or run_phase_from_path(meta_path),
        'status': meta.get('status') or 'unknown',
        'checkpoint': 'yes' if (run_dir / 'best.pt').exists() else 'no',
        'signature': signature,
        'folder': folder,
        'best_epoch': meta.get('best_epoch'),
        'best_valid': meta.get('best_valid_metric'),
        'main_metric': meta.get('main_metric'),
        'ndcg@10': metrics.get('ndcg@10') if isinstance(metrics, dict) else None,
        'hr@10': metrics.get('hr@10') if isinstance(metrics, dict) else None,
        'mrr': metrics.get('mrr') if isinstance(metrics, dict) else None,
        'path': str(run_dir),
        'display_path': display_path,
        'config': config,
        'error': meta.get('error'),
    }


def matches_filters(row: dict, filters: dict):
    config = row.get('config') or {}
    for key, expected in filters.items():
        actual = config.get(key)
        if key == 'data':
            actual = config.get('data') or row.get('data')
        elif key == 'seed':
            actual = config.get('seed') if config.get('seed') is not None else row.get('seed')
        if not values_match(actual, expected):
            return False
    return True


def search(root: Path, filters: dict, *, all_seeds: bool, limit: int | None):
    root = root.resolve()
    rows = [row_from_meta(path, root) for path in iter_meta_paths(root, data=filters.get('data'))]
    rows = [row for row in rows if matches_filters(row, filters)]
    rows.sort(key=lambda row: (str(row.get('data')), str(row.get('model')), str(row.get('signature')), str(row.get('seed'))))
    total_count = len(rows)

    if limit is not None:
        rows = rows[:limit]
    return {
        'filters': filters,
        'count': len(rows),
        'total_count': total_count,
        'runs': rows,
    }


def fmt(value):
    if value is None or value == '':
        return '-'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def print_table(rows: list[dict]):
    columns = [
        ('data', 'data'),
        ('model', 'model'),
        ('repr', 'repr_type'),
        ('task', 'task_type'),
        ('seed', 'seed'),
        ('phase', 'phase'),
        ('status', 'status'),
        ('ckpt', 'checkpoint'),
        ('best', 'best_valid'),
        ('ndcg@10', 'ndcg@10'),
        ('hr@10', 'hr@10'),
        ('sig', 'signature'),
        ('path', 'display_path'),
    ]
    widths = {}
    for title, key in columns:
        values = [fmt(row.get(key)) for row in rows]
        max_value_width = max([len(title), *(len(value) for value in values)], default=len(title))
        widths[key] = min(max_value_width, 64 if key == 'display_path' else 18)

    header = '  '.join(title.ljust(widths[key]) for title, key in columns)
    print(header)
    print('  '.join('-' * widths[key] for _, key in columns))
    for row in rows:
        cells = []
        for _, key in columns:
            value = fmt(row.get(key))
            if len(value) > widths[key]:
                value = value[: widths[key] - 1] + '…'
            cells.append(value.ljust(widths[key]))
        print('  '.join(cells))


def print_report(report: dict):
    total_count = report.get('total_count', report['count'])
    if report['count'] == total_count:
        print(f'trained search matched {total_count} run(s)')
    else:
        print(f'trained search matched {total_count} run(s), showing {report["count"]}')
    if report.get('filters'):
        print('filters=' + json.dumps(report['filters'], sort_keys=True))
    if not report['runs']:
        print('no runs found')
        return
    print_table(report['runs'])


def main():
    parser = argparse.ArgumentParser(description='Search trained artifacts by partial trainer config fields.')
    parser.add_argument('--root', default=str(ROOT), help='Algorithm repository root.')
    parser.add_argument('--data')
    parser.add_argument('--model')
    parser.add_argument('--repr_type', '--repr', dest='repr_type')
    parser.add_argument('--task_type', '--task', dest='task_type')
    parser.add_argument('--repr_source_model')
    parser.add_argument('--sid_export')
    parser.add_argument('--sid_coder')
    parser.add_argument('--hash_coder')
    parser.add_argument('--repr_combine')
    parser.add_argument('--maxitems', type=int)
    parser.add_argument('--model_max_length', type=int)
    parser.add_argument('--item_text_max_tokens', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--accumulate_batch', type=int)
    parser.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float)
    parser.add_argument('--weight_decay', '--wd', dest='weight_decay', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--uid_decoding')
    parser.add_argument('--uid_cluster_levels')
    parser.add_argument('--uid_cluster_topk')
    parser.add_argument('--code_decoding')
    parser.add_argument('--code_beam_width', type=int)
    parser.add_argument('--code_beam_chunk_size', type=int)
    parser.add_argument('--code_collision_loss_weight', type=float)
    parser.add_argument('--main_metric')
    parser.add_argument('--metrics')
    parser.add_argument('--patience', type=int)
    parser.add_argument('--alignment_weight', '--alignment', dest='alignment_weight', type=float)
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--freeze_backbone')
    parser.add_argument('--model_dtype')
    parser.add_argument('--use_lora')
    parser.add_argument('--lora_rank', type=int)
    parser.add_argument('--lora_alpha', type=int)
    parser.add_argument('--lora_dropout', type=float)
    parser.add_argument('--lora_layers')
    parser.add_argument('--lora_target_modules')
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--valid_only')
    parser.add_argument('--test_only')
    parser.add_argument('--overwrite', help='Accepted for copy-pasted trainer commands, ignored by search.')
    parser.add_argument('--load_ckpt', help='Accepted for copy-pasted trainer commands, ignored by search.')
    parser.add_argument('--device', help='Accepted for copy-pasted trainer commands, ignored by search.')
    parser.add_argument('--all-seeds', action='store_true', help='Deprecated no-op; omitted seed already matches all seeds.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON.')
    args = parser.parse_args()

    filters = cli_filters(args)
    try:
        report = search(Path(args.root), filters, all_seeds=args.all_seeds, limit=args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print_report(report)


if __name__ == '__main__':
    main()
