#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUMMARY_COLUMNS = [
    'data',
    'scale_percent',
    'item_count',
    'observed_item_count',
    'user_count',
    'source_user_count',
    'test_user_count',
    'interaction_count',
    'density_percent',
    'sparsity_percent',
]

SUMMARY_HEADERS = {
    'data': 'Data',
    'scale_percent': 'Scale%',
    'item_count': '#Item',
    'observed_item_count': '#ObsItem',
    'user_count': '#User',
    'source_user_count': '#SrcUser',
    'test_user_count': '#TestUser',
    'interaction_count': '#Inter',
    'density_percent': 'Density%',
    'sparsity_percent': 'Sparse%',
}

HISTORY_COLUMNS = [
    'data',
    'history_min',
    'history_p10',
    'history_p25',
    'history_median',
    'history_mean',
    'history_p75',
    'history_p90',
    'history_p95',
    'history_max',
    'history_std',
]


def parse_datasets(raw: str):
    datasets = []
    seen = set()
    for part in str(raw).split(','):
        data = part.strip().lower()
        if not data or data in seen:
            continue
        seen.add(data)
        datasets.append(data)
    if not datasets:
        raise ValueError('--data must contain at least one dataset')
    return datasets


def to_history(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, 'tolist'):
        converted = value.tolist()
        return converted if isinstance(converted, list) else []
    return []


def _quantile(series: pd.Series, value: float):
    if series.empty:
        return None
    return float(series.quantile(value))


def _safe_int(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def summarize_frames(data: str, items: pd.DataFrame, users: pd.DataFrame, test_users, meta: dict):
    item_col = meta.get('item_col') or 'iid'
    user_col = meta.get('user_col') or 'uid'
    history_col = meta.get('history_col') or 'history'
    for column, frame_name in ((item_col, 'items'), (user_col, 'users'), (history_col, 'users')):
        frame = items if frame_name == 'items' else users
        if column not in frame.columns:
            raise ValueError(f'{data}: required column {column!r} missing from {frame_name}.parquet')

    histories = [to_history(value) for value in users[history_col].tolist()]
    lengths = pd.Series([len(history) for history in histories], dtype='float64')
    frequencies = Counter(item for history in histories for item in history)
    item_count = int(items[item_col].nunique())
    user_count = int(users[user_col].nunique())
    interaction_count = int(lengths.sum()) if not lengths.empty else 0
    denominator = user_count * item_count
    density = interaction_count / denominator if denominator else None

    source_user_count = None
    sequences_per_source = None
    if 'source_uid' in users.columns:
        source_user_count = int(users['source_uid'].nunique())
        if source_user_count:
            sequences_per_source = len(users) / source_user_count

    test_user_count = 0
    test_interaction_count = 0
    test_history_mean = None
    if test_users is not None:
        if user_col in test_users.columns:
            test_user_count = int(test_users[user_col].nunique())
        if history_col in test_users.columns:
            test_lengths = pd.Series(
                [len(to_history(value)) for value in test_users[history_col].tolist()],
                dtype='float64',
            )
            test_interaction_count = int(test_lengths.sum()) if not test_lengths.empty else 0
            test_history_mean = float(test_lengths.mean()) if not test_lengths.empty else None

    item_frequencies = pd.Series(list(frequencies.values()), dtype='float64')
    return {
        'data': data,
        'scale_percent': _safe_int(meta.get('scale_percent')),
        'item_count': item_count,
        'observed_item_count': int(len(frequencies)),
        'item_coverage_percent': (100.0 * len(frequencies) / item_count) if item_count else None,
        'user_count': user_count,
        'row_count': int(len(users)),
        'source_user_count': source_user_count,
        'sequences_per_source_user': sequences_per_source,
        'test_user_count': test_user_count,
        'interaction_count': interaction_count,
        'test_interaction_count': test_interaction_count,
        'history_min': _safe_int(lengths.min()) if not lengths.empty else None,
        'history_p10': _quantile(lengths, 0.10),
        'history_p25': _quantile(lengths, 0.25),
        'history_median': _quantile(lengths, 0.50),
        'history_mean': float(lengths.mean()) if not lengths.empty else None,
        'history_p75': _quantile(lengths, 0.75),
        'history_p90': _quantile(lengths, 0.90),
        'history_p95': _quantile(lengths, 0.95),
        'history_max': _safe_int(lengths.max()) if not lengths.empty else None,
        'history_std': float(lengths.std(ddof=0)) if not lengths.empty else None,
        'test_history_mean': test_history_mean,
        'item_frequency_mean': float(item_frequencies.mean()) if not item_frequencies.empty else None,
        'item_frequency_median': _quantile(item_frequencies, 0.50),
        'item_frequency_max': _safe_int(item_frequencies.max()) if not item_frequencies.empty else None,
        'density_percent': 100.0 * density if density is not None else None,
        'sparsity_percent': 100.0 * (1.0 - density) if density is not None else None,
    }


def load_dataset_summary(root: Path, data: str, prepare: bool):
    formatted_dir = root / 'artifacts' / 'formatted' / data
    required = [formatted_dir / 'items.parquet', formatted_dir / 'users.parquet', formatted_dir / 'meta.json']
    if prepare and not all(path.exists() for path in required):
        from utils.pipeline import ensure_formatted

        ensure_formatted(data)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f'{data}: formatted artifacts are missing: {", ".join(missing)}. '
            f'Run `python formatter.py --data {data}` or omit --no-prepare.'
        )

    meta = json.loads((formatted_dir / 'meta.json').read_text())
    items = pd.read_parquet(formatted_dir / 'items.parquet')
    users = pd.read_parquet(formatted_dir / 'users.parquet')
    test_path = formatted_dir / 'test_users.parquet'
    test_users = pd.read_parquet(test_path) if test_path.exists() else None
    return summarize_frames(data, items, users, test_users, meta)


def format_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f'{value:,.1f}'
        if abs(value) >= 10:
            return f'{value:.2f}'
        return f'{value:.4f}'
    return str(value)


def render_table(records, columns, headers=None):
    headers = headers or {}
    labels = [headers.get(column, column) for column in columns]
    rows = [[format_value(record.get(column)) for column in columns] for record in records]
    widths = [
        max(len(labels[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    lines = ['  '.join(label.ljust(widths[index]) for index, label in enumerate(labels))]
    lines.append('  '.join('-' * width for width in widths))
    for row in rows:
        lines.append('  '.join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return '\n'.join(lines)


def render_markdown(records, columns):
    def cell(value):
        return format_value(value).replace('|', '\\|')

    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
    ]
    for record in records:
        lines.append('| ' + ' | '.join(cell(record.get(column)) for column in columns) + ' |')
    return '\n'.join(lines) + '\n'


def write_output(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        pd.DataFrame(records).to_csv(path, index=False)
    elif suffix in {'.md', '.markdown'}:
        path.write_text(render_markdown(records, list(records[0])))
    elif suffix == '.json':
        path.write_text(json.dumps(records, indent=2) + '\n')
    else:
        raise ValueError('--output extension must be .csv, .md, .markdown, or .json')


def main():
    parser = argparse.ArgumentParser(
        description='Compare common statistics for one or more formatted recommendation datasets.'
    )
    parser.add_argument('--data', required=True, help='Comma-separated datasets, e.g. ras1,ras2,ras5,ras99.')
    parser.add_argument('--root', type=Path, default=ROOT, help='Path to the secommenders-algorithm repository.')
    parser.add_argument('--output', type=Path, default=None, help='Optional .csv, .md, or .json output path.')
    parser.add_argument(
        '--no-prepare',
        action='store_true',
        help='Do not automatically run the formatter when a dataset artifact is missing.',
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_path = args.output.resolve() if args.output is not None else None
    os.chdir(root)
    try:
        datasets = parse_datasets(args.data)
        records = [load_dataset_summary(root, data, prepare=not args.no_prepare) for data in datasets]
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print('\nDataset Summary')
    print(render_table(records, SUMMARY_COLUMNS, SUMMARY_HEADERS))
    print('\nUser History Length Distribution')
    print(render_table(records, HISTORY_COLUMNS))

    if output_path is not None:
        try:
            write_output(output_path, records)
        except ValueError as exc:
            parser.error(str(exc))
        print(f'\nwrote full statistics to {output_path}')


if __name__ == '__main__':
    main()
