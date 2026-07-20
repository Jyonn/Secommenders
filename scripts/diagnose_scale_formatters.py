import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def to_list(value):
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def quantiles(values, qs=(0.1, 0.5, 0.9)):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {f'p{int(q * 100)}': None for q in qs}
    output = {}
    for q in qs:
        position = (len(values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            value = values[lower]
        else:
            weight = position - lower
            value = values[lower] * (1.0 - weight) + values[upper] * weight
        output[f'p{int(q * 100)}'] = float(value)
    return output


def fmt(value):
    if value is None:
        return '-'
    if isinstance(value, float):
        if math.isnan(value):
            return '-'
        return f'{value:.6g}'
    if isinstance(value, int):
        return f'{value:,}'
    return str(value)


def print_table(title, headers, rows):
    print()
    print(title)
    rows = [[fmt(cell) for cell in row] for row in rows]
    widths = []
    for idx, header in enumerate(headers):
        widths.append(max(len(header), *[len(row[idx]) for row in rows], 1))
    print('  '.join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        print('  '.join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def dataset_dirs(root: Path, data: str):
    data = data.lower()
    return {
        'formatted': root / 'artifacts' / 'formatted' / data,
        'processed': root / 'artifacts' / 'processed' / data,
    }


def load_frame(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path)


def history_lengths(frame, history_col):
    if frame is None or history_col not in frame:
        return []
    return [len(to_list(value)) for value in frame[history_col].tolist()]


def iter_histories(frame, history_col):
    if frame is None or history_col not in frame:
        return
    for value in frame[history_col].tolist():
        history = to_list(value)
        if history:
            yield history


def count_histories(frame, history_col):
    counter = Counter()
    total = 0
    for history in iter_histories(frame, history_col):
        counter.update(history)
        total += len(history)
    return counter, total


def gini_from_counts(counts):
    if not counts:
        return None
    values = sorted(float(value) for value in counts)
    total = sum(values)
    if total <= 0:
        return None
    n = len(values)
    weighted_sum = sum(index * value for index, value in enumerate(values, start=1))
    return float((2.0 * weighted_sum / (n * total)) - ((n + 1.0) / n))


def entropy_effective_items(counter: Counter):
    total = sum(counter.values())
    if total <= 0:
        return None
    entropy = -sum((value / total) * math.log(max(value / total, 1e-300)) for value in counter.values())
    return float(math.exp(entropy))


def top_share(counter: Counter, ratio: float):
    total = sum(counter.values())
    if total <= 0 or not counter:
        return None
    topn = max(1, int(math.ceil(len(counter) * ratio)))
    return sum(value for _, value in counter.most_common(topn)) / float(total)


def target_items(frame, history_col, multi_item_col=None):
    targets = []
    if frame is None:
        return targets
    if multi_item_col and multi_item_col in frame:
        for value in frame[multi_item_col].tolist():
            targets.extend(to_list(value))
        return targets
    if history_col not in frame:
        return targets
    for history in iter_histories(frame, history_col):
        targets.append(history[-1])
    return targets


def summarize_dataset(root: Path, data: str):
    dirs = dataset_dirs(root, data)
    formatted_dir = dirs['formatted']
    processed_dir = dirs['processed']
    formatted_meta = read_json(formatted_dir / 'meta.json')
    processed_meta = read_json(processed_dir / 'meta.json')
    meta = processed_meta or formatted_meta

    history_col = meta.get('history_col') or 'history'
    user_col = meta.get('user_col') or 'uid'
    multi_item_col = meta.get('multi_item_col')

    formatted_users = load_frame(formatted_dir / 'users.parquet')
    formatted_test = load_frame(formatted_dir / 'test_users.parquet')
    processed_items = load_frame(processed_dir / 'items.parquet')
    finetune = load_frame(processed_dir / 'finetune.parquet')
    valid = load_frame(processed_dir / 'valid.parquet')
    test = load_frame(processed_dir / 'test.parquet')

    formatted_lengths = history_lengths(formatted_users, history_col)
    finetune_lengths = history_lengths(finetune, history_col)
    valid_lengths = history_lengths(valid, history_col)
    test_lengths = history_lengths(test, history_col)

    source_uid_stats = {}
    if formatted_users is not None and 'source_uid' in formatted_users:
        counts = formatted_users['source_uid'].astype(str).value_counts()
        source_uid_stats = {
            'source_users': int(len(counts)),
            'segments_per_source_mean': float(counts.mean()) if len(counts) else None,
            'segments_per_source_p90': float(counts.quantile(0.9)) if len(counts) else None,
            'sources_with_multi_segments': int((counts > 1).sum()),
        }

    train_counter, train_interactions = count_histories(finetune, history_col)
    valid_counter, valid_interactions = count_histories(valid, history_col)
    test_counter, test_interactions = count_histories(test, history_col)
    all_public_counter = train_counter + valid_counter + test_counter

    targets = target_items(test, history_col, multi_item_col=multi_item_col)
    target_counter = Counter(targets)
    target_train_freq = [train_counter.get(item, 0) for item in targets]

    output = {
        'data': data,
        'exists_formatted': formatted_users is not None,
        'exists_processed': finetune is not None,
        'version': formatted_meta.get('version'),
        'policy': formatted_meta.get('small_scale_policy') or 'chunked-multi-sequence',
        'scale_percent': formatted_meta.get('scale_percent'),
        'scale_test_ratio': formatted_meta.get('scale_test_ratio'),
        'scale_total_sequences': formatted_meta.get('scale_total_sequences'),
        'scale_train_limit': formatted_meta.get('scale_train_limit'),
        'scale_test_start': formatted_meta.get('scale_test_start'),
        'formatted_sequences': None if formatted_users is None else int(len(formatted_users)),
        'formatted_test_sequences': None if formatted_test is None else int(len(formatted_test)),
        'processed_items': None if processed_items is None else int(len(processed_items)),
        'finetune_sequences': None if finetune is None else int(len(finetune)),
        'valid_sequences': None if valid is None else int(len(valid)),
        'test_sequences': None if test is None else int(len(test)),
        'formatted_len_mean': float(sum(formatted_lengths) / len(formatted_lengths)) if formatted_lengths else None,
        'formatted_len_p50': quantiles(formatted_lengths).get('p50'),
        'formatted_len_p90': quantiles(formatted_lengths).get('p90'),
        'finetune_len_mean': float(sum(finetune_lengths) / len(finetune_lengths)) if finetune_lengths else None,
        'finetune_len_p50': quantiles(finetune_lengths).get('p50'),
        'finetune_len_p90': quantiles(finetune_lengths).get('p90'),
        'valid_len_mean': float(sum(valid_lengths) / len(valid_lengths)) if valid_lengths else None,
        'test_len_mean': float(sum(test_lengths) / len(test_lengths)) if test_lengths else None,
        'train_interactions': int(train_interactions),
        'train_prediction_count': int(sum(max(length - 1, 0) for length in finetune_lengths)),
        'valid_interactions': int(valid_interactions),
        'test_interactions': int(test_interactions),
        'train_unique_items': int(len(train_counter)),
        'public_unique_items': int(len(all_public_counter)),
        'train_freq_mean': float(sum(train_counter.values()) / len(train_counter)) if train_counter else None,
        'train_freq_p50': quantiles(list(train_counter.values())).get('p50'),
        'train_freq_p90': quantiles(list(train_counter.values())).get('p90'),
        'train_freq_gini': gini_from_counts(train_counter.values()),
        'train_effective_items': entropy_effective_items(train_counter),
        'train_top1pct_share': top_share(train_counter, 0.01),
        'train_top5pct_share': top_share(train_counter, 0.05),
        'test_targets': int(len(targets)),
        'test_target_unique_items': int(len(target_counter)),
        'test_target_top1pct_share': top_share(target_counter, 0.01),
        'test_target_top5pct_share': top_share(target_counter, 0.05),
        'test_target_seen_rate': None if not targets else sum(1 for item in targets if train_counter.get(item, 0) > 0) / len(targets),
        'test_target_train_freq_mean': (
            float(sum(target_train_freq) / len(target_train_freq)) if target_train_freq else None
        ),
        'test_target_train_freq_p50': quantiles(target_train_freq).get('p50'),
        'test_target_train_freq_p90': quantiles(target_train_freq).get('p90'),
    }
    output.update(source_uid_stats)
    return output


def compare_pairs(summaries):
    by_data = {summary['data']: summary for summary in summaries}
    rows = []
    for data, left in by_data.items():
        if data.startswith('ra') and not data.startswith('ras'):
            right = by_data.get('ras' + data[2:])
            if right:
                rows.append((data, right['data'], left, right))
        if data.startswith('rv') and not data.startswith('rvs'):
            right = by_data.get('rvs' + data[2:])
            if right:
                rows.append((data, right['data'], left, right))
    return rows


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose RecIF scale formatter statistics for RA/RAS/RV/RVS comparisons.'
    )
    parser.add_argument('--root', default=str(ROOT), help='Path to secommenders-algorithm root.')
    parser.add_argument(
        '--data',
        required=True,
        help='Comma-separated datasets, e.g. ra1,ras1,ra5,ras5,ra20,ras20.',
    )
    parser.add_argument('--json-output', default=None, help='Optional path to write full JSON summary.')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets = [part.strip().lower() for part in args.data.split(',') if part.strip()]
    summaries = [summarize_dataset(root, data) for data in datasets]

    missing = [summary['data'] for summary in summaries if not summary['exists_processed']]
    if missing:
        print(
            'warning: processed artifacts missing for '
            + ', '.join(missing)
            + '. Run formatter.py and processor.py for those datasets first.'
        )

    print_table(
        'Split / Sequence Summary',
        [
            'data',
            'policy',
            'scale',
            'test_ratio',
            'fmt_seq',
            'train_seq',
            'test_seq',
            'items',
            'src_users',
            'seg/src',
            'multi_src',
            'train_len',
            'preds',
        ],
        [
            [
                s['data'],
                'latest' if s['policy'] != 'chunked-multi-sequence' else 'chunked',
                s['scale_percent'],
                s['scale_test_ratio'],
                s['formatted_sequences'],
                s['finetune_sequences'],
                s['test_sequences'],
                s['processed_items'],
                s.get('source_users'),
                s.get('segments_per_source_mean'),
                s.get('sources_with_multi_segments'),
                s['finetune_len_mean'],
                s['train_prediction_count'],
            ]
            for s in summaries
        ],
    )

    print_table(
        'Train Item Frequency / Headness',
        [
            'data',
            'train_items',
            'interactions',
            'freq_mean',
            'freq_p50',
            'freq_p90',
            'gini',
            'eff_items',
            'top1%',
            'top5%',
        ],
        [
            [
                s['data'],
                s['train_unique_items'],
                s['train_interactions'],
                s['train_freq_mean'],
                s['train_freq_p50'],
                s['train_freq_p90'],
                s['train_freq_gini'],
                s['train_effective_items'],
                s['train_top1pct_share'],
                s['train_top5pct_share'],
            ]
            for s in summaries
        ],
    )

    print_table(
        'Test Target Concentration',
        [
            'data',
            'targets',
            'target_items',
            'seen_rate',
            'target_freq_mean',
            'target_freq_p50',
            'target_freq_p90',
            'target_top1%',
            'target_top5%',
        ],
        [
            [
                s['data'],
                s['test_targets'],
                s['test_target_unique_items'],
                s['test_target_seen_rate'],
                s['test_target_train_freq_mean'],
                s['test_target_train_freq_p50'],
                s['test_target_train_freq_p90'],
                s['test_target_top1pct_share'],
                s['test_target_top5pct_share'],
            ]
            for s in summaries
        ],
    )

    pair_rows = []
    for left_name, right_name, left, right in compare_pairs(summaries):
        pair_rows.append(
            [
                f'{left_name}->{right_name}',
                ratio(right['formatted_sequences'], left['formatted_sequences']),
                ratio(right['finetune_sequences'], left['finetune_sequences']),
                ratio(right['train_prediction_count'], left['train_prediction_count']),
                ratio(right['train_interactions'], left['train_interactions']),
                ratio(right['train_unique_items'], left['train_unique_items']),
                ratio(right['test_target_train_freq_mean'], left['test_target_train_freq_mean']),
                ratio(right['test_target_top5pct_share'], left['test_target_top5pct_share']),
            ]
        )
    if pair_rows:
        print_table(
            'Paired Formatter Ratios',
            [
                'pair',
                'fmt_seq',
                'train_seq',
                'preds',
                'interactions',
                'train_items',
                'target_freq',
                'target_top5%',
            ],
            pair_rows,
        )

    if args.json_output:
        output_path = Path(args.json_output)
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + '\n')
        print(f'\nwrote JSON summary to {output_path}')


if __name__ == '__main__':
    main()
