import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ARTIFACT_ROOT = Path('artifacts')


def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def human_bytes(size: int):
    value = float(size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024.0 or unit == 'TB':
            return f'{value:.1f}{unit}' if unit != 'B' else f'{int(value)}B'
        value /= 1024.0
    return f'{size}B'


def as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, 'tolist'):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    try:
        return list(value)
    except TypeError:
        return []


def is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def safe_len(value):
    if is_missing(value):
        return 0
    if isinstance(value, str):
        return len(value)
    if hasattr(value, '__len__'):
        return len(value)
    return 0


def fmt_number(value, digits=3):
    if value is None:
        return '-'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        if math.isnan(value):
            return '-'
        if abs(value) >= 1000:
            return f'{value:,.{digits}f}'
        return f'{value:.{digits}f}'.rstrip('0').rstrip('.')
    return str(value)


def pct(part, total):
    if not total:
        return 0.0
    return 100.0 * float(part) / float(total)


def print_title(title: str):
    print()
    print('=' * 100)
    print(title)
    print('=' * 100)


def print_section(title: str):
    print()
    print(f'-- {title} --')


def print_kv(rows):
    width = max((len(str(key)) for key, _ in rows), default=0)
    for key, value in rows:
        print(f'  {str(key).rjust(width)} : {value}')


def print_table(headers, rows, max_width=36):
    rows = [[str(cell) for cell in row] for row in rows]
    headers = [str(header) for header in headers]
    widths = []
    for index, header in enumerate(headers):
        values = [row[index] for row in rows] if rows else []
        width = max([len(header), *[len(value) for value in values]], default=len(header))
        widths.append(min(width, max_width))

    def trim(value, width):
        if len(value) <= width:
            return value
        if width <= 3:
            return value[:width]
        return value[:width - 3] + '...'

    print('  ' + '  '.join(trim(header, widths[i]).ljust(widths[i]) for i, header in enumerate(headers)))
    print('  ' + '  '.join('-' * width for width in widths))
    for row in rows:
        print('  ' + '  '.join(trim(row[i], widths[i]).ljust(widths[i]) for i in range(len(headers))))


def describe_numeric(values):
    series = pd.Series(values)
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return {}
    quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    return {
        'count': int(series.shape[0]),
        'sum': float(series.sum()),
        'mean': float(series.mean()),
        'std': float(series.std(ddof=0)),
        'min': float(series.min()),
        'p25': float(quantiles.get(0.25, 0.0)),
        'p50': float(quantiles.get(0.5, 0.0)),
        'p75': float(quantiles.get(0.75, 0.0)),
        'p90': float(quantiles.get(0.9, 0.0)),
        'p95': float(quantiles.get(0.95, 0.0)),
        'p99': float(quantiles.get(0.99, 0.0)),
        'max': float(series.max()),
    }


def print_numeric_summary(title, values):
    stats = describe_numeric(values)
    if not stats:
        print(f'  {title}: no numeric values')
        return stats
    print(f'  {title}:')
    print_kv(
        [
            ('count', fmt_number(stats['count'])),
            ('sum', fmt_number(stats['sum'])),
            ('mean', fmt_number(stats['mean'])),
            ('std', fmt_number(stats['std'])),
            ('min', fmt_number(stats['min'])),
            ('p25', fmt_number(stats['p25'])),
            ('p50', fmt_number(stats['p50'])),
            ('p75', fmt_number(stats['p75'])),
            ('p90', fmt_number(stats['p90'])),
            ('p95', fmt_number(stats['p95'])),
            ('p99', fmt_number(stats['p99'])),
            ('max', fmt_number(stats['max'])),
        ]
    )
    return stats


def histogram_bins(values, bins):
    series = pd.Series(values)
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return []
    min_value = float(series.min())
    max_value = float(series.max())
    if min_value == max_value:
        return [(min_value, max_value, int(series.shape[0]))]
    counts = pd.cut(series, bins=bins, include_lowest=True).value_counts(sort=False)
    result = []
    for interval, count in counts.items():
        result.append((float(interval.left), float(interval.right), int(count)))
    return result


def print_histogram(title, values, *, bins=20, width=48):
    hist = histogram_bins(values, bins)
    if not hist:
        print(f'  {title}: no histogram values')
        return
    max_count = max(count for _, _, count in hist) or 1
    print(f'  {title}:')
    for left, right, count in hist:
        bar_len = int(round(width * count / max_count))
        bar = '#' * bar_len
        print(f'    [{left:>8.2f}, {right:>8.2f}] {str(count).rjust(8)} | {bar}')


def parse_count_bucket_spec(spec: str | None):
    if not spec:
        return None
    buckets = []
    tokens = [token.strip() for token in spec.replace('/', ',').split(',') if token.strip()]
    for token in tokens:
        if token.endswith('+'):
            left_text = token[:-1].strip()
            if not left_text:
                raise ValueError(f'invalid bucket token: {token!r}')
            left = int(left_text)
            buckets.append((left, None, f'{left}+'))
            continue
        if '-' in token:
            left_text, right_text = [part.strip() for part in token.split('-', 1)]
            left = int(left_text)
            right = int(right_text)
            if right < left:
                raise ValueError(f'invalid bucket token with descending range: {token!r}')
            buckets.append((left, right, f'{left}-{right}'))
            continue
        value = int(token)
        buckets.append((value, value, str(value)))
    if not buckets:
        return None
    return buckets


def count_bucket_histogram(values, bucket_spec):
    if not bucket_spec:
        return []
    series = pd.Series(values)
    series = pd.to_numeric(series, errors='coerce').dropna().astype(int)
    if series.empty:
        return []
    hist = []
    total_matched = 0
    for left, right, label in bucket_spec:
        if right is None:
            mask = series >= left
        else:
            mask = (series >= left) & (series <= right)
        count = int(mask.sum())
        total_matched += count
        hist.append((label, count))
    unmatched = int(series.shape[0]) - total_matched
    if unmatched:
        hist.append(('unmatched', unmatched))
    return hist


def print_count_bucket_histogram(title, values, *, bucket_spec, width=48):
    hist = count_bucket_histogram(values, bucket_spec)
    if not hist:
        print(f'  {title}: no histogram values')
        return hist
    max_count = max(count for _, count in hist) or 1
    print(f'  {title}:')
    for label, count in hist:
        bar_len = int(round(width * count / max_count))
        bar = '#' * bar_len
        print(f'    {label.rjust(12)} {str(count).rjust(8)} | {bar}')
    return hist


def top_counter(counter: Counter, topk: int):
    return counter.most_common(topk)


def load_frame(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path)


def frame_memory_mb(frame: pd.DataFrame):
    return float(frame.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)


def file_table(stage_dir: Path):
    rows = []
    for path in sorted(stage_dir.glob('*')):
        if not path.is_file():
            continue
        rows.append([path.name, human_bytes(path.stat().st_size)])
    return rows


def infer_columns(meta: dict, items: pd.DataFrame | None, frames: dict[str, pd.DataFrame]):
    item_col = meta.get('item_col')
    user_col = meta.get('user_col')
    history_col = meta.get('history_col')

    if item_col is None and items is not None and len(items.columns):
        item_col = items.columns[0]
    sample_frame = next((frame for frame in frames.values() if frame is not None and len(frame.columns)), None)
    if sample_frame is not None:
        if user_col is None:
            user_col = sample_frame.columns[0]
        if history_col is None:
            candidates = [column for column in sample_frame.columns if str(column).lower() in {'history', 'hist', 'hist_pid'}]
            history_col = candidates[0] if candidates else (sample_frame.columns[1] if len(sample_frame.columns) > 1 else None)
    return item_col, user_col, history_col


def analyze_columns(frame: pd.DataFrame, *, title: str, topk: int):
    print_section(f'{title} columns')
    rows = []
    total = len(frame)
    for column in frame.columns:
        series = frame[column]
        null_count = int(series.isna().sum())
        unique_count = None
        try:
            unique_count = int(series.nunique(dropna=True))
        except TypeError:
            unique_count = '-'
        sample = None
        for value in series.head(50):
            if not is_missing(value):
                sample = value
                break
        sample_text = repr(sample)
        rows.append(
            [
                column,
                str(series.dtype),
                fmt_number(null_count),
                f'{pct(null_count, total):.2f}%',
                fmt_number(unique_count) if isinstance(unique_count, int) else str(unique_count),
                sample_text[:80],
            ]
        )
    print_table(['column', 'dtype', 'nulls', 'null%', 'unique', 'sample'], rows)

    text_rows = []
    for column in frame.columns:
        series = frame[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        text_lengths = non_null.map(lambda value: len(str(value)))
        stats = describe_numeric(text_lengths)
        if not stats:
            continue
        text_rows.append(
            [
                column,
                fmt_number(stats['mean']),
                fmt_number(stats['p50']),
                fmt_number(stats['p95']),
                fmt_number(stats['max']),
            ]
        )
    if text_rows:
        print_section(f'{title} text-like column lengths')
        print_table(['column', 'mean_len', 'p50', 'p95', 'max'], text_rows)


def collect_history_stats(frame: pd.DataFrame, history_col: str, item_counter: Counter):
    lengths = []
    unique_lengths = []
    repeat_events = 0
    rows_with_repeats = 0
    empty_rows = 0
    for history in frame[history_col].tolist():
        values = [str(item) for item in as_list(history)]
        length = len(values)
        unique_length = len(set(values))
        lengths.append(length)
        unique_lengths.append(unique_length)
        if length == 0:
            empty_rows += 1
        if length > unique_length:
            rows_with_repeats += 1
            repeat_events += length - unique_length
        item_counter.update(values)
    return {
        'lengths': lengths,
        'unique_lengths': unique_lengths,
        'repeat_events': repeat_events,
        'rows_with_repeats': rows_with_repeats,
        'empty_rows': empty_rows,
    }


def collect_multi_item_counter(frame: pd.DataFrame, column: str | None):
    counter = Counter()
    if not column or column not in frame.columns:
        return counter
    for values in frame[column].tolist():
        counter.update(str(item) for item in as_list(values))
    return counter


def analyze_user_frame(
    name: str,
    frame: pd.DataFrame,
    *,
    item_col: str | None,
    user_col: str | None,
    history_col: str | None,
    item_ids: set[str],
    multi_item_col: str | None,
    bins: int,
    width: int,
    topk: int,
    popularity_bucket_spec=None,
):
    print_section(f'{name} rows')
    rows = [
        ('rows', fmt_number(len(frame))),
        ('memory', f'{frame_memory_mb(frame):.2f}MB'),
    ]
    if user_col and user_col in frame.columns:
        duplicate_users = int(frame.duplicated(subset=[user_col]).sum())
        rows.extend(
            [
                ('user_col', user_col),
                ('unique_users', fmt_number(frame[user_col].nunique(dropna=True))),
                ('duplicate_user_rows', fmt_number(duplicate_users)),
            ]
        )
    if history_col and history_col in frame.columns:
        item_counter = Counter()
        history_stats = collect_history_stats(frame, history_col, item_counter)
        lengths = history_stats['lengths']
        unique_lengths = history_stats['unique_lengths']
        observed_items = set(item_counter)
        missing_items = observed_items - item_ids if item_ids else set()
        rows.extend(
            [
                ('history_col', history_col),
                ('total_interactions', fmt_number(sum(lengths))),
                ('unique_history_items', fmt_number(len(observed_items))),
                ('item_catalog_coverage', f'{pct(len(observed_items & item_ids), len(item_ids)):.2f}%' if item_ids else '-'),
                ('missing_history_items', fmt_number(len(missing_items))),
                ('empty_histories', fmt_number(history_stats['empty_rows'])),
                ('rows_with_repeated_items', fmt_number(history_stats['rows_with_repeats'])),
                ('repeated_item_events', fmt_number(history_stats['repeat_events'])),
            ]
        )
        multi_counter = collect_multi_item_counter(frame, multi_item_col)
        if multi_counter:
            multi_items = set(multi_counter)
            rows.extend(
                [
                    (f'{multi_item_col}_events', fmt_number(sum(multi_counter.values()))),
                    (f'{multi_item_col}_unique_items', fmt_number(len(multi_items))),
                    (f'{multi_item_col}_missing_items', fmt_number(len(multi_items - item_ids)) if item_ids else '-'),
                ]
            )
        print_kv(rows)
        print_numeric_summary(f'{name} history length summary', lengths)
        print_histogram(f'{name} history length histogram', lengths, bins=bins, width=width)
        print_numeric_summary(f'{name} unique-history-item count summary', unique_lengths)
        popularity_values = list(item_counter.values())
        print_numeric_summary(f'{name} item popularity summary', popularity_values)
        popularity_histogram = None
        if popularity_bucket_spec:
            popularity_histogram = print_count_bucket_histogram(
                f'{name} item popularity histogram',
                popularity_values,
                bucket_spec=popularity_bucket_spec,
                width=width,
            )
        else:
            print_histogram(f'{name} item popularity histogram', popularity_values, bins=bins, width=width)
        top_rows = [[item, fmt_number(count)] for item, count in top_counter(item_counter, topk)]
        if top_rows:
            print_section(f'{name} top {topk} items by history frequency')
            print_table(['item', 'count'], top_rows)
        if missing_items:
            print_section(f'{name} missing item examples')
            print('  ' + ', '.join(sorted(list(missing_items))[:topk]))
        return {
            'rows': len(frame),
            'total_interactions': int(sum(lengths)),
            'unique_history_items': int(len(observed_items)),
            'missing_history_items': int(len(missing_items)),
            'history_length': describe_numeric(lengths),
            'item_popularity': describe_numeric(popularity_values),
            'item_popularity_buckets': popularity_histogram,
            'top_items': top_counter(item_counter, topk),
        }

    print_kv(rows)
    return {'rows': len(frame)}


def analyze_item_frame(items: pd.DataFrame, item_col: str | None, *, topk: int, bins: int, width: int):
    print_section('items')
    rows = [
        ('rows', fmt_number(len(items))),
        ('columns', fmt_number(len(items.columns))),
        ('memory', f'{frame_memory_mb(items):.2f}MB'),
    ]
    item_ids = set()
    if item_col and item_col in items.columns:
        duplicate_items = int(items.duplicated(subset=[item_col]).sum())
        null_items = int(items[item_col].isna().sum())
        item_ids = set(str(item) for item in items[item_col].dropna().tolist())
        rows.extend(
            [
                ('item_col', item_col),
                ('unique_items', fmt_number(len(item_ids))),
                ('duplicate_item_rows', fmt_number(duplicate_items)),
                ('null_item_ids', fmt_number(null_items)),
            ]
        )
    print_kv(rows)
    analyze_columns(items, title='items', topk=topk)

    text_columns = [
        column for column in items.columns
        if column != item_col and (pd.api.types.is_object_dtype(items[column]) or pd.api.types.is_string_dtype(items[column]))
    ]
    for column in text_columns:
        lengths = items[column].dropna().map(lambda value: len(str(value))).tolist()
        if lengths:
            print_histogram(f'items.{column} string length histogram', lengths, bins=bins, width=width)
    return item_ids


def overlap_report(frames: dict[str, pd.DataFrame], user_col: str | None, history_col: str | None):
    print_section('split overlap')
    split_names = list(frames)
    rows = []
    if user_col:
        user_sets = {
            name: set(frame[user_col].dropna().map(str).tolist())
            for name, frame in frames.items()
            if user_col in frame.columns
        }
        for i, left in enumerate(split_names):
            for right in split_names[i + 1:]:
                if left in user_sets and right in user_sets:
                    rows.append([f'{left} vs {right}', 'users', fmt_number(len(user_sets[left] & user_sets[right]))])
    if history_col:
        item_sets = {}
        for name, frame in frames.items():
            if history_col not in frame.columns:
                continue
            values = set()
            for history in frame[history_col].tolist():
                values.update(str(item) for item in as_list(history))
            item_sets[name] = values
        for i, left in enumerate(split_names):
            for right in split_names[i + 1:]:
                if left in item_sets and right in item_sets:
                    rows.append([f'{left} vs {right}', 'history_items', fmt_number(len(item_sets[left] & item_sets[right]))])
    if rows:
        print_table(['pair', 'type', 'overlap'], rows)
    else:
        print('  no comparable split overlap data')


def analyze_stage(data: str, stage: str, *, bins: int, width: int, topk: int, popularity_bucket_spec=None):
    stage_dir = ARTIFACT_ROOT / stage / data
    if not stage_dir.exists():
        raise FileNotFoundError(f'{stage} artifact directory not found: {stage_dir}')

    print_title(f'{stage.upper()} ANALYSIS: {data}')
    meta = read_json(stage_dir / 'meta.json')
    stats = read_json(stage_dir / 'stats.json')
    if meta:
        print_section('meta')
        print_kv([(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value) for key, value in meta.items()])
    if stats:
        print_section('existing stats.json')
        print_kv([(key, fmt_number(value) if isinstance(value, (int, float)) else value) for key, value in stats.items()])

    print_section('artifact files')
    print_table(['file', 'size'], file_table(stage_dir))

    if stage == 'formatted':
        item_path = stage_dir / 'items.parquet'
        frames = {
            'users': load_frame(stage_dir / 'users.parquet'),
            'test_users': load_frame(stage_dir / 'test_users.parquet'),
        }
    elif stage == 'processed':
        item_path = stage_dir / 'items.parquet'
        frames = {
            'finetune': load_frame(stage_dir / 'finetune.parquet'),
            'valid': load_frame(stage_dir / 'valid.parquet'),
            'test': load_frame(stage_dir / 'test.parquet'),
        }
    else:
        raise ValueError(f'unsupported stage: {stage}')

    frames = {name: frame for name, frame in frames.items() if frame is not None}
    items = load_frame(item_path)
    item_col, user_col, history_col = infer_columns(meta, items, frames)
    multi_item_col = meta.get('multi_item_col')

    print_section('resolved schema')
    print_kv(
        [
            ('item_col', item_col or '-'),
            ('user_col', user_col or '-'),
            ('history_col', history_col or '-'),
            ('multi_item_col', multi_item_col or '-'),
        ]
    )

    item_ids = set()
    report = {'stage': stage, 'data': data, 'meta': meta, 'stats': stats, 'frames': {}}
    if items is not None:
        item_ids = analyze_item_frame(items, item_col, topk=topk, bins=bins, width=width)
        report['item_count'] = len(items)
        report['unique_items'] = len(item_ids)
    else:
        print_section('items')
        print(f'  missing {item_path}')

    for name, frame in frames.items():
        analyze_columns(frame, title=name, topk=topk)
        report['frames'][name] = analyze_user_frame(
            name,
            frame,
            item_col=item_col,
            user_col=user_col,
            history_col=history_col,
            item_ids=item_ids,
            multi_item_col=multi_item_col,
            bins=bins,
            width=width,
            topk=topk,
            popularity_bucket_spec=popularity_bucket_spec,
        )
    overlap_report(frames, user_col, history_col)
    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze formatted or processed Secommenders dataset artifacts.')
    parser.add_argument('--data', required=True, help='Dataset name, e.g. mind or recifadsall.')
    parser.add_argument('--stage', default='formatted', choices=['formatted', 'processed', 'all'])
    parser.add_argument('--bins', type=int, default=20, help='Histogram bin count.')
    parser.add_argument('--width', type=int, default=48, help='Histogram bar width.')
    parser.add_argument('--topk', type=int, default=20, help='Top-K frequent values to show.')
    parser.add_argument(
        '--popularity-buckets',
        default=None,
        help='Custom item popularity buckets, e.g. "1/2/3/4/5/6/7/8/9/10+" or "1,2,3,4,5-9,10+".',
    )
    parser.add_argument('--json-out', default=None, help='Optional path to save machine-readable summary JSON.')
    args = parser.parse_args()

    popularity_bucket_spec = parse_count_bucket_spec(args.popularity_buckets)
    stages = ['formatted', 'processed'] if args.stage == 'all' else [args.stage]
    reports = {}
    for stage in stages:
        reports[stage] = analyze_stage(
            args.data.lower(),
            stage,
            bins=args.bins,
            width=args.width,
            topk=args.topk,
            popularity_bucket_spec=popularity_bucket_spec,
        )

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + '\n')
        print()
        print(f'wrote summary json to {path}')


if __name__ == '__main__':
    main()
