#!/usr/bin/env python3

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_ROOT = ROOT / 'artifacts' / 'scheduler'


def sanitize_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_') or 'scheduler'


def load_manifest(plan_path: Path):
    import yaml

    plan = yaml.safe_load(plan_path.read_text()) or {}
    plan_name = sanitize_name(plan.get('name') or plan_path.stem)
    manifest_path = ARTIFACT_ROOT / plan_name / 'frequency_breakdown_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'frequency manifest not found: {manifest_path}; '
            f'run scripts/scheduler_frequency_breakdown.py first'
        )
    return json.loads(manifest_path.read_text()), manifest_path


def representation_addons(args):
    parts = [part.strip() for part in str(args.get('repr_type') or '').split('+') if part.strip()]
    return tuple(sorted(part for part in parts if part not in {'uid', 'sid'}))


def pairing_key(args):
    addons = representation_addons(args)
    semantic_source = args.get('repr_source_model') if 'embedding' in addons else None
    return (
        str(args.get('data')),
        str(args.get('model')),
        addons,
        semantic_source,
    )


def rank_metrics(rank, k):
    if rank is None:
        return 0.0, 0.0, 0.0
    rank = int(rank)
    hit = 1.0 if rank <= k else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    return hit, ndcg, 1.0 / rank


def load_item_ranking(path, k):
    payload = json.loads(Path(path).read_text())
    records = payload.get('records')
    if not records:
        raise ValueError(
            f'{path} has no per-target records; rerun scheduler_frequency_breakdown.py '
            f'with the updated trainer'
        )
    rows = []
    for record in records:
        raw_item_id = record.get('raw_item_id')
        if raw_item_id is None:
            raise ValueError(f'{path} contains records without raw_item_id; rerun the analysis')
        hr, ndcg, mrr = rank_metrics(record.get('rank'), k)
        rows.append({
            'item_id': str(raw_item_id),
            'frequency': int(record['frequency']),
            'frequency_bucket': str(record['frequency_bucket']),
            f'hr@{k}': hr,
            f'ndcg@{k}': ndcg,
            'mrr': mrr,
        })
    frame = pd.DataFrame(rows)
    return frame.groupby('item_id', as_index=False).agg({
        'frequency': 'max',
        'frequency_bucket': 'first',
        f'hr@{k}': 'mean',
        f'ndcg@{k}': 'mean',
        'mrr': 'mean',
    })


def resolve_transfer_path(template, data):
    if '{data}' in template:
        return Path(template.format(data=data))
    path = Path(template)
    if path.is_dir() or not path.suffix:
        return path / f'{data}.parquet'
    return path


def read_transfer_quality(path):
    suffix = path.suffix.lower()
    if suffix == '.csv':
        frame = pd.read_csv(path)
    elif suffix == '.json':
        frame = pd.read_json(path)
    else:
        frame = pd.read_parquet(path)
    frame['item_id'] = frame['item_id'].astype(str)
    return frame


def pair_experiments(entries):
    grouped = defaultdict(lambda: {'uid': [], 'sid': []})
    for entry in entries:
        task_type = str((entry.get('args') or {}).get('task_type'))
        if task_type in {'uid', 'sid'}:
            grouped[pairing_key(entry['args'])][task_type].append(entry)
    pairs = []
    for key, methods in grouped.items():
        for uid_entry in methods['uid']:
            for sid_entry in methods['sid']:
                pairs.append((key, uid_entry, sid_entry))
    return pairs


def balanced_tie_preserving_threshold(values):
    """Choose the most balanced binary split without separating equal values."""
    counts = values.value_counts().sort_index()
    if len(counts) < 2:
        return None
    cumulative = counts.cumsum()
    total = int(counts.sum())
    candidates = cumulative.iloc[:-1]
    split_position = (candidates - (total - candidates)).abs().argmin()
    return float(counts.index[split_position])


def assign_transfer_quality(joined, *, tq_column, tq_split):
    joined = joined.copy()
    if tq_split == 'global':
        threshold = float(joined[tq_column].median())
        joined['tq_threshold'] = threshold
    elif tq_split == 'within-bucket':
        thresholds = joined.groupby('frequency_bucket', sort=False)[tq_column].transform(
            balanced_tie_preserving_threshold
        )
        joined['tq_threshold'] = thresholds
    else:
        raise ValueError(f'unsupported transfer-quality split: {tq_split!r}')

    joined = joined.dropna(subset=['tq_threshold']).copy()
    joined['transfer_quality'] = (
        joined[tq_column] > joined['tq_threshold']
    ).map({True: 'high', False: 'low'})
    return joined


def summarize_pair(
    uid_frame,
    sid_frame,
    transfer,
    *,
    tq_column,
    k,
    tq_split='within-bucket',
):
    joined = uid_frame.merge(
        sid_frame,
        on=['item_id', 'frequency', 'frequency_bucket'],
        suffixes=('_uid', '_sid'),
        validate='one_to_one',
    ).merge(
        transfer[['item_id', tq_column]],
        on='item_id',
        validate='one_to_one',
    )
    joined = joined.dropna(subset=[tq_column]).copy()
    if joined.empty:
        return joined, pd.DataFrame()
    joined = assign_transfer_quality(
        joined,
        tq_column=tq_column,
        tq_split=tq_split,
    )
    metric_names = [f'hr@{k}', f'ndcg@{k}', 'mrr']
    rows = []
    for (frequency_bucket, transfer_quality), group in joined.groupby(
        ['frequency_bucket', 'transfer_quality'],
        sort=False,
    ):
        row = {
            'frequency_bucket': frequency_bucket,
            'transfer_quality': transfer_quality,
            'item_count': len(group),
            'tq_mean': float(group[tq_column].mean()),
            'tq_threshold': float(group['tq_threshold'].iloc[0]),
            'tq_split': tq_split,
        }
        for metric in metric_names:
            uid_value = float(group[f'{metric}_uid'].mean())
            sid_value = float(group[f'{metric}_sid'].mean())
            row[f'uid_{metric}'] = uid_value
            row[f'sid_{metric}'] = sid_value
            row[f'delta_{metric}'] = sid_value - uid_value
        rows.append(row)
    return joined, pd.DataFrame(rows)


def print_summary(label, summary, k):
    print(f'\n=== {label} ===')
    if summary.empty:
        print('no joined items')
        return
    columns = [
        'frequency_bucket',
        'transfer_quality',
        'item_count',
        'tq_mean',
        'tq_threshold',
        f'uid_hr@{k}',
        f'sid_hr@{k}',
        f'delta_hr@{k}',
        f'delta_ndcg@{k}',
        'delta_mrr',
    ]
    print(summary[columns].to_string(index=False, float_format=lambda value: f'{value:.4f}'))


def method_label(entry):
    args = entry['args']
    if args.get('task_type') == 'sid':
        return (
            f'{args.get("repr_type")}/'
            f'{args.get("sid_coder", "sid")}/'
            f'{args.get("sid_export", "default")}'
        )
    return str(args.get('repr_type'))


def main():
    parser = argparse.ArgumentParser(
        description='Join UID/SID target ranks with per-item transfer quality and frequency.'
    )
    parser.add_argument('--plan', required=True)
    parser.add_argument(
        '--transfer-quality',
        required=True,
        help='Per-item transfer output path or template containing {data}.',
    )
    parser.add_argument('--tq-column', default='content_ndcg@20')
    parser.add_argument(
        '--tq-split',
        choices=['within-bucket', 'global'],
        default='within-bucket',
        help=(
            'How to form high/low Transfer Quality groups. The default chooses '
            'a balanced, tie-preserving threshold separately within each '
            'frequency bucket; global reproduces the legacy whole-pair median.'
        ),
    )
    parser.add_argument('--metric-k', type=int, default=10)
    parser.add_argument('--output', default=None, help='Optional combined CSV output.')
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    manifest, manifest_path = load_manifest(plan_path)
    pairs = pair_experiments(manifest.get('experiments') or [])
    print(
        f'manifest={manifest_path} comparable_pairs={len(pairs)} '
        f'tq_split={args.tq_split}'
    )
    all_summaries = []
    transfer_cache = {}
    for key, uid_entry, sid_entry in pairs:
        data, model, addons, _ = key
        transfer_path = resolve_transfer_path(args.transfer_quality, data)
        if data not in transfer_cache:
            if not transfer_path.exists():
                print(f'\nskip data={data}: transfer quality not found at {transfer_path}')
                continue
            transfer_cache[data] = read_transfer_quality(transfer_path)
        transfer = transfer_cache[data]
        if args.tq_column not in transfer:
            raise ValueError(f'{transfer_path} does not contain column {args.tq_column!r}')
        uid_frame = load_item_ranking(uid_entry['analysis_path'], args.metric_k)
        sid_frame = load_item_ranking(sid_entry['analysis_path'], args.metric_k)
        _, summary = summarize_pair(
            uid_frame,
            sid_frame,
            transfer,
            tq_column=args.tq_column,
            k=args.metric_k,
            tq_split=args.tq_split,
        )
        label = (
            f'data={data} model={model} addons={"+".join(addons) or "none"} '
            f'UID={method_label(uid_entry)} SID={method_label(sid_entry)}'
        )
        print_summary(label, summary, args.metric_k)
        if not summary.empty:
            summary.insert(0, 'sid_experiment', sid_entry['name'])
            summary.insert(0, 'uid_experiment', uid_entry['name'])
            summary.insert(0, 'addons', '+'.join(addons) or 'none')
            summary.insert(0, 'model', model)
            summary.insert(0, 'data', data)
            all_summaries.append(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
        combined.to_csv(output_path, index=False)
        print(f'\nwrote combined analysis to {output_path}')


if __name__ == '__main__':
    main()
