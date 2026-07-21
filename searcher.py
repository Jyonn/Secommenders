import argparse
import json
import shlex
import sys
from pathlib import Path

from utils.artifact_identity import normalize_legacy_train_config


ROOT = Path(__file__).resolve().parent

RUNTIME_KEYS = {
    'root',
    'json',
    'all_seeds',
    'limit',
    'interactive',
    'no_interactive',
    'results_only',
    'show_common',
    'overwrite',
    'load_ckpt',
    'device',
}
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
    test_metrics = meta.get('test_metrics') if isinstance(meta.get('test_metrics'), dict) else None
    valid_metrics = meta.get('valid_metrics') if isinstance(meta.get('valid_metrics'), dict) else None
    metrics = test_metrics if test_metrics is not None else valid_metrics or {}
    metric_source = 'test' if test_metrics is not None else 'valid' if valid_metrics is not None else '-'
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
        'has_result': 'yes' if metrics else 'no',
        'metric_source': metric_source,
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
        'metrics': metrics,
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
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return ','.join(fmt(item) for item in value)
    return str(value)


BASE_COLUMNS = [
    ('#', '_row_id'),
    ('data', 'data'),
    ('model', 'model'),
    ('repr', 'repr_type'),
    ('task', 'task_type'),
    ('seed', 'seed'),
    ('phase', 'phase'),
    ('status', 'status'),
    ('result', 'has_result'),
    ('src', 'metric_source'),
    ('ckpt', 'checkpoint'),
    ('best', 'best_valid'),
    ('ndcg@10', 'ndcg@10'),
    ('hr@10', 'hr@10'),
    ('mrr', 'mrr'),
    ('sig', 'signature'),
    ('path', 'display_path'),
]

ALWAYS_SHOW_KEYS = {'_row_id', 'display_path'}


def row_value(row: dict, key: str):
    if key in row:
        return row.get(key)
    return (row.get('config') or {}).get(key)


def common_values(rows: list[dict], keys: list[str]):
    common = {}
    for key in keys:
        values = [normalize_for_compare(row_value(row, key)) for row in rows]
        if values and all(value == values[0] for value in values):
            common[key] = row_value(rows[0], key)
    return common


def display_columns(rows: list[dict], *, show_common: bool = False):
    columns = list(BASE_COLUMNS)
    if show_common or len(rows) <= 1:
        return columns
    common = common_values(rows, [key for _, key in columns if key not in ALWAYS_SHOW_KEYS])
    return [(title, key) for title, key in columns if key in ALWAYS_SHOW_KEYS or key not in common]


def _clip(value: str, width: int):
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + '…'


def _print_common(rows: list[dict], columns: list[tuple[str, str]], *, show_common: bool):
    if show_common or len(rows) <= 1:
        return
    hidden_keys = [key for _, key in BASE_COLUMNS if key not in {key for _, key in columns} and key not in ALWAYS_SHOW_KEYS]
    common = common_values(rows, hidden_keys)
    if not common:
        return
    parts = [f'{key}={fmt(value)}' for key, value in common.items() if fmt(value) != '-']
    if not parts:
        return
    preview = ', '.join(parts[:10])
    if len(parts) > 10:
        preview += f', ... +{len(parts) - 10}'
    print(f'common: {preview}')


def print_table(rows: list[dict], *, show_common: bool = False):
    columns = display_columns(rows, show_common=show_common)
    for index, row in enumerate(rows, start=1):
        row.setdefault('_row_id', index)
    _print_common(rows, columns, show_common=show_common)
    if not rows:
        print('no runs found')
        return
    if not columns:
        columns = [('#', '_row_id'), ('path', 'display_path')]
    widths = {}
    for title, key in columns:
        values = [fmt(row_value(row, key)) for row in rows]
        max_value_width = max([len(title), *(len(value) for value in values)], default=len(title))
        widths[key] = min(max_value_width, 72 if key == 'display_path' else 18)

    header = '  '.join(title.ljust(widths[key]) for title, key in columns)
    print(header)
    print('  '.join('-' * widths[key] for _, key in columns))
    for row in rows:
        cells = []
        for _, key in columns:
            value = _clip(fmt(row_value(row, key)), widths[key])
            cells.append(value.ljust(widths[key]))
        print('  '.join(cells))


def print_legacy_table(rows: list[dict]):
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


def print_report(report: dict, *, show_common: bool = False):
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
    print_table(report['runs'], show_common=show_common)


def has_result(row: dict):
    return bool(row.get('metrics')) or row.get('has_result') == 'yes'


def parse_id_list(text: str, max_id: int):
    ids = []
    for token in text.replace(',', ' ').split():
        if '-' in token:
            start_text, end_text = token.split('-', 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f'invalid id range: {token}')
            start, end = int(start_text), int(end_text)
            step = 1 if start <= end else -1
            ids.extend(range(start, end + step, step))
        else:
            if not token.isdigit():
                raise ValueError(f'invalid id: {token}')
            ids.append(int(token))
    bad = [item for item in ids if item < 1 or item > max_id]
    if bad:
        raise ValueError(f'id out of range: {bad[0]}')
    deduped = []
    seen = set()
    for item in ids:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def select_rows(rows: list[dict], spec: str):
    ids = parse_id_list(spec, len(rows))
    return [rows[index - 1] for index in ids]


def filter_rows_by_expr(rows: list[dict], expr: str):
    if '=' not in expr:
        raise ValueError('filter expression must be key=value')
    key, value = expr.split('=', 1)
    key = key.strip()
    expected = coerce_scalar(value.strip())
    if not key:
        raise ValueError('filter key is empty')
    return [row for row in rows if values_match(row_value(row, key), expected)]


def metric_keys(rows: list[dict]):
    keys = set()
    preferred = ['loss', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'hr@5', 'hr@10', 'hr@20', 'mrr']
    for row in rows:
        metrics = row.get('metrics') if isinstance(row.get('metrics'), dict) else {}
        keys.update(metrics)
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def config_keys(rows: list[dict]):
    keys = set()
    for row in rows:
        keys.update((row.get('config') or {}).keys())
    preferred = [key for key in FILTER_KEYS if key in keys]
    return preferred + sorted(keys - set(preferred))


def print_kv_table(pairs: list[tuple[str, object]], *, title: str | None = None):
    if title:
        print(title)
    if not pairs:
        print('  -')
        return
    width = min(max(len(key) for key, _ in pairs), 36)
    for key, value in pairs:
        print(f'  {key.ljust(width)} : {fmt(value)}')


def print_single_detail(row: dict):
    print(f'run #{row.get("_row_id", "-")} {row.get("display_path")}')
    meta_pairs = [
        ('status', row.get('status')),
        ('result', row.get('has_result')),
        ('metric_source', row.get('metric_source')),
        ('checkpoint', row.get('checkpoint')),
        ('signature', row.get('signature')),
        ('folder', row.get('folder')),
        ('best_epoch', row.get('best_epoch')),
        ('best_valid', row.get('best_valid')),
        ('main_metric', row.get('main_metric')),
        ('path', row.get('path')),
    ]
    print_kv_table(meta_pairs, title='meta')
    print_kv_table(list((row.get('metrics') or {}).items()), title='metrics')
    config = row.get('config') or {}
    print_kv_table([(key, config.get(key)) for key in config_keys([row])], title='config')
    if row.get('error'):
        print('error:')
        print(row['error'])


def print_comparison(rows: list[dict], *, show_common: bool = False):
    if len(rows) == 1:
        print_single_detail(rows[0])
        return
    labels = [f'#{row.get("_row_id", index + 1)}' for index, row in enumerate(rows)]
    fields = [
        ('data', lambda row: row.get('data')),
        ('model', lambda row: row.get('model')),
        ('repr_type', lambda row: row.get('repr_type')),
        ('task_type', lambda row: row.get('task_type')),
        ('seed', lambda row: row.get('seed')),
        ('phase', lambda row: row.get('phase')),
        ('status', lambda row: row.get('status')),
        ('result', lambda row: row.get('has_result')),
        ('signature', lambda row: row.get('signature')),
        ('best_epoch', lambda row: row.get('best_epoch')),
        ('best_valid', lambda row: row.get('best_valid')),
    ]
    fields.extend((f'metric.{key}', lambda row, key=key: (row.get('metrics') or {}).get(key)) for key in metric_keys(rows))
    for key in config_keys(rows):
        fields.append((f'config.{key}', lambda row, key=key: (row.get('config') or {}).get(key)))
    if not show_common:
        fields = [
            (name, getter) for name, getter in fields
            if len({json.dumps(normalize_for_compare(getter(row)), sort_keys=True) for row in rows}) > 1
        ]
    if not fields:
        print('selected runs have no differing displayed fields')
        return
    first_width = min(max(len(name) for name, _ in fields), 34)
    value_widths = []
    for row_index, _ in enumerate(rows):
        values = [fmt(getter(rows[row_index])) for _, getter in fields]
        value_widths.append(min(max([len(labels[row_index]), *(len(value) for value in values)]), 28))
    header = 'field'.ljust(first_width) + '  ' + '  '.join(label.ljust(value_widths[index]) for index, label in enumerate(labels))
    print(header)
    print('-' * len(header))
    for name, getter in fields:
        values = [_clip(fmt(getter(row)), value_widths[index]) for index, row in enumerate(rows)]
        print(name.ljust(first_width) + '  ' + '  '.join(value.ljust(value_widths[index]) for index, value in enumerate(values)))


def print_help():
    print(
        '\n'.join(
            [
                'commands:',
                '  <ids>                 compare runs, e.g. 1 or 1,2,5 or 3-8',
                '  keep <ids>            narrow current table to selected run numbers',
                '  where key=value       filter current table by row/config field',
                '  results               show only runs with valid/test metrics',
                '  all                   restore original matched runs',
                '  common                toggle hidden common columns',
                '  table                 redraw current table',
                '  help                  show this help',
                '  quit                  exit',
            ]
        )
    )


def interactive_loop(report: dict, *, show_common: bool = False, results_only: bool = False):
    original = list(report['runs'])
    rows = [row for row in original if has_result(row)] if results_only else list(original)
    common = show_common

    def redraw():
        for index, row in enumerate(rows, start=1):
            row['_row_id'] = index
        print(f'\nshowing {len(rows)} / {len(original)} matched run(s)')
        print_table(rows, show_common=common)

    redraw()
    print('type help for commands; type run numbers such as "1,2" to compare')
    while True:
        try:
            command = input('searcher> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not command:
            continue
        lower = command.lower()
        try:
            if lower in {'q', 'quit', 'exit'}:
                return
            if lower in {'h', 'help', '?'}:
                print_help()
                continue
            if lower in {'t', 'table', 'ls'}:
                redraw()
                continue
            if lower in {'common', 'toggle common'}:
                common = not common
                redraw()
                continue
            if lower in {'results', 'result', 'has-result'}:
                rows = [row for row in rows if has_result(row)]
                redraw()
                continue
            if lower in {'all', 'reset'}:
                rows = list(original)
                redraw()
                continue
            if lower.startswith('keep '):
                rows = select_rows(rows, command.split(maxsplit=1)[1])
                redraw()
                continue
            if lower.startswith('where '):
                rows = filter_rows_by_expr(rows, command.split(maxsplit=1)[1])
                redraw()
                continue
            if lower.startswith('compare '):
                print_comparison(select_rows(rows, command.split(maxsplit=1)[1]), show_common=common)
                continue
            tokens = shlex.split(command)
            if tokens and all(token.replace(',', '').replace('-', '').isdigit() for token in tokens):
                print_comparison(select_rows(rows, command), show_common=common)
                continue
            print('unknown command; type help')
        except ValueError as exc:
            print(f'error: {exc}')


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
    parser.add_argument('--interactive', action='store_true', help='Open an interactive search/refine prompt after matching runs.')
    parser.add_argument('--no-interactive', action='store_true', help='Print once and exit even when running in a terminal.')
    parser.add_argument('--results-only', action='store_true', help='Only show runs that contain valid/test metrics.')
    parser.add_argument('--show-common', action='store_true', help='Do not hide columns whose values are common to all displayed runs.')
    args = parser.parse_args()

    filters = cli_filters(args)
    try:
        report = search(
            Path(args.root),
            filters,
            all_seeds=args.all_seeds,
            limit=None if args.results_only else args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.results_only:
        result_runs = [row for row in report['runs'] if has_result(row)]
        report['total_count'] = len(result_runs)
        report['runs'] = result_runs[:args.limit] if args.limit is not None else result_runs
        report['count'] = len(report['runs'])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    wants_interactive = args.interactive or (
        not args.no_interactive
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    if wants_interactive:
        interactive_loop(report, show_common=args.show_common, results_only=False)
    else:
        print_report(report, show_common=args.show_common)


if __name__ == '__main__':
    main()
