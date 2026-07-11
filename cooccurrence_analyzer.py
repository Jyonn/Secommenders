import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path('artifacts')


def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def as_list(value):
    if value is None or isinstance(value, str):
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


def fmt(value, digits=4):
    if value is None:
        return '-'
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        if math.isnan(value):
            return '-'
        return f'{value:.{digits}f}'.rstrip('0').rstrip('.')
    return str(value)


def print_title(title):
    print()
    print('=' * 100)
    print(title)
    print('=' * 100)


def print_section(title):
    print()
    print(f'-- {title} --')


def print_kv(rows):
    width = max((len(str(key)) for key, _ in rows), default=0)
    for key, value in rows:
        print(f'  {str(key).rjust(width)} : {value}')


def print_table(headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max([len(str(headers[index])), *[len(row[index]) for row in rows]], default=len(str(headers[index])))
        for index in range(len(headers))
    ]
    print('  ' + '  '.join(str(headers[index]).ljust(widths[index]) for index in range(len(headers))))
    print('  ' + '  '.join('-' * width for width in widths))
    for row in rows:
        print('  ' + '  '.join(row[index].ljust(widths[index]) for index in range(len(headers))))


def print_histogram(title, buckets, *, width=48):
    if not buckets:
        return
    print_section(title)
    max_count = max(count for _, count, _ in buckets) or 1
    for label, count, value in buckets:
        bar = '#' * int(round(width * count / max_count))
        print(f'  {label.rjust(18)} {str(count).rjust(8)} {fmt(value).rjust(10)} | {bar}')


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def load_processed(data: str, split_names: list[str]):
    directory = ARTIFACT_ROOT / 'processed' / data
    meta = read_json(directory / 'meta.json')
    if not meta:
        raise FileNotFoundError(f'processed meta not found: {directory / "meta.json"}')
    item_col = meta['item_col']
    history_col = meta['history_col']
    items = pd.read_parquet(directory / 'items.parquet')
    item_ids = [str(item) for item in items[item_col].tolist()]
    frames = {}
    for split in split_names:
        path = directory / f'{split}.parquet'
        if path.exists():
            frames[split] = pd.read_parquet(path)
    if not frames:
        raise FileNotFoundError(f'no processed split files found in {directory}: {split_names}')
    return meta, item_ids, frames, history_col


def iter_window_pairs(history, item_to_index, window):
    values = [item_to_index.get(str(item)) for item in as_list(history)]
    values = [value for value in values if value is not None]
    for center_pos, center in enumerate(values):
        left = max(0, center_pos - window)
        right = min(len(values), center_pos + window + 1)
        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue
            context = values[context_pos]
            if center == context:
                continue
            yield (center, context) if center < context else (context, center)


def build_cooccurrence(frames, history_col, item_to_index, *, window, max_pairs):
    pair_counts = Counter()
    item_counts = Counter()
    total_events = 0
    total_pairs = 0
    histories = 0
    for frame in frames.values():
        for history in frame[history_col].tolist():
            indices = [item_to_index.get(str(item)) for item in as_list(history)]
            indices = [index for index in indices if index is not None]
            item_counts.update(indices)
            total_events += len(indices)
            histories += 1
            for pair in iter_window_pairs(history, item_to_index, window):
                pair_counts[pair] += 1
                total_pairs += 1
                if max_pairs and total_pairs >= max_pairs:
                    return pair_counts, item_counts, total_events, histories
    return pair_counts, item_counts, total_events, histories


def compute_ppmi(pair_counts, item_counts, total_events):
    total_pair_events = sum(pair_counts.values())
    scores = {}
    if total_pair_events <= 0 or total_events <= 0:
        return scores
    for (left, right), count in pair_counts.items():
        p_ij = count / total_pair_events
        p_i = item_counts[left] / total_events
        p_j = item_counts[right] / total_events
        if p_i <= 0 or p_j <= 0:
            continue
        pmi = math.log(p_ij / (p_i * p_j))
        scores[(left, right)] = max(0.0, pmi)
    return scores


def load_embedding(data: str, model: str, item_ids: list[str]):
    directory = ARTIFACT_ROOT / 'embedded' / data / model
    embeddings_path = directory / 'embeddings.npy'
    item_ids_path = directory / 'item_ids.parquet'
    if not embeddings_path.exists() or not item_ids_path.exists():
        raise FileNotFoundError(f'embedded artifact not found: {directory}')
    embeddings = np.load(embeddings_path).astype(np.float32)
    ids_frame = pd.read_parquet(item_ids_path)
    id_col = ids_frame.columns[0]
    embedding_ids = [str(item) for item in ids_frame[id_col].tolist()]
    index = {item: pos for pos, item in enumerate(embedding_ids)}
    missing = [item for item in item_ids if item not in index]
    if missing:
        raise ValueError(f'embeddings missing {len(missing)} processed items; first={missing[:5]}')
    ordered = np.asarray([embeddings[index[item]] for item in item_ids], dtype=np.float32)
    return normalize_rows(ordered)


def resolve_quantized_export(data: str, embedding_model: str, quantizer: str, export: str):
    root = ARTIFACT_ROOT / 'quantized' / data
    candidates = []
    for directory in root.glob('*'):
        export_dir = directory / 'exports' / export
        meta_path = export_dir / 'meta.json'
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        if str(meta.get('embedding_model')) == embedding_model and str(meta.get('quantizer_model')) == quantizer:
            candidates.append(export_dir)
    legacy = root / embedding_model / quantizer / 'exports' / export
    if (legacy / 'meta.json').exists():
        candidates.append(legacy)
    if not candidates:
        raise FileNotFoundError(
            f'quantized export not found for data={data} embedding_model={embedding_model} '
            f'quantizer={quantizer} export={export}'
        )
    candidates = sorted(set(candidates), key=lambda path: str(path))
    return candidates[-1]


def load_sid_codes(data: str, embedding_model: str, quantizer: str, export: str, item_ids: list[str]):
    export_dir = resolve_quantized_export(data, embedding_model, quantizer, export)
    codes_path = export_dir / 'codebook_indices.npy'
    item_ids_path = export_dir / 'item_ids.parquet'
    if not codes_path.exists() or not item_ids_path.exists():
        raise FileNotFoundError(f'SID export is missing codes or item ids: {export_dir}')
    codes = np.load(codes_path)
    if codes.ndim == 1:
        codes = codes.reshape(-1, 1)
    ids_frame = pd.read_parquet(item_ids_path)
    id_col = ids_frame.columns[0]
    code_ids = [str(item) for item in ids_frame[id_col].tolist()]
    index = {item: pos for pos, item in enumerate(code_ids)}
    missing = [item for item in item_ids if item not in index]
    if missing:
        raise ValueError(f'SID codes missing {len(missing)} processed items; first={missing[:5]}')
    ordered = np.asarray([codes[index[item]] for item in item_ids])
    return ordered, export_dir


def pair_similarity_from_embeddings(embeddings, pairs):
    left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    return np.sum(embeddings[left] * embeddings[right], axis=1)


def sid_prefix_length(left, right):
    total = 0
    for a, b in zip(left, right):
        if a != b:
            break
        total += 1
    return total


def sid_pair_features(codes, pairs):
    prefix = []
    hamming = []
    for left, right in pairs:
        left_code = codes[left]
        right_code = codes[right]
        prefix.append(sid_prefix_length(left_code, right_code))
        hamming.append(int(np.sum(left_code != right_code)))
    return np.asarray(prefix), np.asarray(hamming)


def auc_from_scores(pos_scores, neg_scores):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    scores = pd.Series(np.concatenate([pos_scores, neg_scores]))
    ranks = scores.rank(method='average').to_numpy(dtype=np.float64)
    pos_ranks = ranks[:len(pos_scores)]
    auc = (pos_ranks.sum() - len(pos_scores) * (len(pos_scores) + 1) / 2.0) / (len(pos_scores) * len(neg_scores))
    return float(auc)


def sample_negative_pairs(num_items, positives, item_counts, count, rng):
    positive_set = set(positives)
    items = np.arange(num_items, dtype=np.int64)
    weights = np.asarray([item_counts.get(int(item), 0) for item in items], dtype=np.float64)
    if weights.sum() <= 0:
        weights = None
    else:
        weights /= weights.sum()
    negatives = []
    attempts = 0
    max_attempts = max(count * 50, 1000)
    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        if weights is None:
            left, right = rng.sample(range(num_items), 2)
        else:
            left, right = rng.choices(items.tolist(), weights=weights.tolist(), k=2)
            if left == right:
                continue
        pair = (left, right) if left < right else (right, left)
        if pair in positive_set:
            continue
        negatives.append(pair)
    return negatives


def describe_scores(name, pos_scores, neg_scores, *, higher_is_closer=True):
    if pos_scores is None or neg_scores is None or len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    auc = auc_from_scores(pos_scores, neg_scores) if higher_is_closer else auc_from_scores(-pos_scores, -neg_scores)
    return {
        'name': name,
        'positive_mean': float(np.mean(pos_scores)),
        'negative_mean': float(np.mean(neg_scores)),
        'delta': float(np.mean(pos_scores) - np.mean(neg_scores)),
        'positive_p50': float(np.quantile(pos_scores, 0.5)),
        'negative_p50': float(np.quantile(neg_scores, 0.5)),
        'auc': auc,
    }


def bucket_by_similarity(pos_scores, neg_scores, bins=10):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    if len(scores) == 0:
        return []
    edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 1:
        return []
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & (scores <= right if right == edges[-1] else scores < right)
        count = int(mask.sum())
        if count <= 0:
            continue
        positive_rate = float(labels[mask].mean())
        rows.append((f'{left:.3f}..{right:.3f}', count, positive_rate))
    return rows


def build_anchor_relevance(pair_scores):
    relevance = defaultdict(dict)
    for (left, right), score in pair_scores.items():
        relevance[left][right] = score
        relevance[right][left] = score
    return relevance


def dcg(values):
    total = 0.0
    for index, value in enumerate(values):
        total += float(value) / math.log2(index + 2)
    return total


def retrieval_metrics_from_scorer(score_anchor_fn, num_items, relevance, *, topks, max_anchors, rng):
    anchors = [anchor for anchor, rel in relevance.items() if rel]
    if max_anchors and len(anchors) > max_anchors:
        anchors = rng.sample(anchors, max_anchors)
    rows = []
    max_topk = max(topks)
    for topk in topks:
        ndcgs = []
        recalls = []
        hits = []
        for anchor in anchors:
            rel = relevance.get(anchor) or {}
            if not rel:
                continue
            scores = score_anchor_fn(anchor).copy()
            scores[anchor] = -np.inf
            candidate_count = min(max_topk, num_items - 1)
            if candidate_count <= 0:
                continue
            nearest = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
            nearest = nearest[np.argsort(-scores[nearest])][:topk]
            gains = [rel.get(int(item), 0.0) for item in nearest]
            ideal = sorted(rel.values(), reverse=True)[:topk]
            ideal_dcg = dcg(ideal)
            ndcgs.append(dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0)
            relevant_top = set(sorted(rel, key=rel.get, reverse=True)[:topk])
            recalls.append(len(set(int(item) for item in nearest) & relevant_top) / max(len(relevant_top), 1))
            hits.append(1.0 if any(value > 0 for value in gains) else 0.0)
        rows.append(
            {
                'topk': topk,
                'anchors': len(anchors),
                'ndcg': float(np.mean(ndcgs)) if ndcgs else 0.0,
                'recall': float(np.mean(recalls)) if recalls else 0.0,
                'hit_rate': float(np.mean(hits)) if hits else 0.0,
            }
        )
    return rows


def sid_anchor_similarity(codes, anchor):
    return np.mean(codes == codes[anchor], axis=1).astype(np.float32)


def print_score_report(rows):
    print_section('positive vs negative pair similarity')
    table = []
    for row in rows:
        if row is None:
            continue
        table.append(
            [
                row['name'],
                fmt(row['positive_mean']),
                fmt(row['negative_mean']),
                fmt(row['delta']),
                fmt(row['positive_p50']),
                fmt(row['negative_p50']),
                fmt(row['auc']),
            ]
        )
    if table:
        print_table(['space', 'pos_mean', 'neg_mean', 'delta', 'pos_p50', 'neg_p50', 'auc'], table)
    else:
        print('  no comparable scores')


def print_retrieval_report(name, rows):
    print_section(f'{name} nearest-neighbor retrieval of co-occurrence')
    print_table(
        ['topk', 'anchors', 'ndcg', 'recall', 'hit_rate'],
        [[row['topk'], row['anchors'], fmt(row['ndcg']), fmt(row['recall']), fmt(row['hit_rate'])] for row in rows],
    )


def main():
    parser = argparse.ArgumentParser(description='Analyze whether item content/SID similarity aligns with sequence co-occurrence.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--splits', default='finetune', help='Comma-separated processed splits, e.g. finetune,valid.')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--min-item-freq', type=int, default=5)
    parser.add_argument('--max-pairs', type=int, default=2_000_000)
    parser.add_argument('--sample-pairs', type=int, default=100_000)
    parser.add_argument('--negative-per-positive', type=int, default=1)
    parser.add_argument('--embedding-model', default=None, help='Content embedding model under artifacts/embedded/<data>/<model>.')
    parser.add_argument(
        '--sid-embedding-model',
        default=None,
        help='Optional override for SID embedding model. Defaults to --embedding-model.',
    )
    parser.add_argument('--sid-coder', default=None)
    parser.add_argument('--sid-export', default='coll')
    parser.add_argument('--topk', default='20,50,100')
    parser.add_argument('--max-anchors', type=int, default=2000)
    parser.add_argument('--buckets', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json-out', default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = args.data.lower()
    split_names = [part.strip() for part in args.splits.split(',') if part.strip()]
    topks = [int(part.strip()) for part in args.topk.split(',') if part.strip()]

    meta, item_ids, frames, history_col = load_processed(data, split_names)
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    pair_counts, item_counts, total_events, histories = build_cooccurrence(
        frames,
        history_col,
        item_to_index,
        window=args.window,
        max_pairs=args.max_pairs,
    )
    eligible_items = {item for item, count in item_counts.items() if count >= args.min_item_freq}
    pair_counts = Counter(
        {
            pair: count for pair, count in pair_counts.items()
            if pair[0] in eligible_items and pair[1] in eligible_items
        }
    )
    ppmi = compute_ppmi(pair_counts, item_counts, total_events)

    positive_pairs = list(pair_counts)
    if args.sample_pairs and len(positive_pairs) > args.sample_pairs:
        positive_pairs = rng.sample(positive_pairs, args.sample_pairs)
    negative_count = max(len(positive_pairs) * args.negative_per_positive, 1)
    negative_pairs = sample_negative_pairs(len(item_ids), positive_pairs, item_counts, negative_count, rng)

    print_title(f'CO-OCCURRENCE ANALYSIS: {data}')
    print_kv(
        [
            ('splits', ','.join(frames.keys())),
            ('history_col', history_col),
            ('items', fmt(len(item_ids))),
            ('histories', fmt(histories)),
            ('window', fmt(args.window)),
            ('total_item_events', fmt(total_events)),
            ('cooccurrence_pairs', fmt(len(pair_counts))),
            ('eligible_items', fmt(len(eligible_items))),
            ('positive_sample_pairs', fmt(len(positive_pairs))),
            ('negative_sample_pairs', fmt(len(negative_pairs))),
        ]
    )

    reports = {
        'data': data,
        'splits': list(frames.keys()),
        'window': args.window,
        'item_count': len(item_ids),
        'history_count': histories,
        'total_item_events': total_events,
        'cooccurrence_pair_count': len(pair_counts),
        'eligible_item_count': len(eligible_items),
        'spaces': {},
    }

    score_rows = []
    relevance = build_anchor_relevance(ppmi)

    if args.embedding_model:
        embeddings = load_embedding(data, args.embedding_model, item_ids)
        pos_scores = pair_similarity_from_embeddings(embeddings, positive_pairs)
        neg_scores = pair_similarity_from_embeddings(embeddings, negative_pairs)
        row = describe_scores(f'embedding:{args.embedding_model}', pos_scores, neg_scores, higher_is_closer=True)
        if row:
            score_rows.append(row)
            reports['spaces'][f'embedding:{args.embedding_model}'] = row
        print_histogram(
            f'embedding:{args.embedding_model} positive-rate by similarity bucket',
            bucket_by_similarity(pos_scores, neg_scores, bins=args.buckets),
        )
        retrieval_rows = retrieval_metrics_from_scorer(
            lambda anchor: embeddings @ embeddings[anchor],
            len(item_ids),
            relevance,
            topks=topks,
            max_anchors=args.max_anchors,
            rng=rng,
        )
        print_retrieval_report(f'embedding:{args.embedding_model}', retrieval_rows)
        reports['spaces'].setdefault(f'embedding:{args.embedding_model}', {})['retrieval'] = retrieval_rows

    sid_embedding_model = args.sid_embedding_model or args.embedding_model
    if sid_embedding_model and args.sid_coder:
        codes, export_dir = load_sid_codes(data, sid_embedding_model, args.sid_coder, args.sid_export, item_ids)
        pos_prefix, pos_hamming = sid_pair_features(codes, positive_pairs)
        neg_prefix, neg_hamming = sid_pair_features(codes, negative_pairs)
        prefix_row = describe_scores(f'sid-prefix:{args.sid_coder}/{args.sid_export}', pos_prefix, neg_prefix, higher_is_closer=True)
        hamming_row = describe_scores(f'sid-hamming:{args.sid_coder}/{args.sid_export}', pos_hamming, neg_hamming, higher_is_closer=False)
        score_rows.extend(row for row in [prefix_row, hamming_row] if row)
        if prefix_row:
            reports['spaces'][prefix_row['name']] = prefix_row
        if hamming_row:
            reports['spaces'][hamming_row['name']] = hamming_row
        print_section('SID export')
        print_kv([('export_dir', export_dir), ('code_shape', list(codes.shape))])
        print_histogram('SID prefix positive-rate bucket', bucket_by_similarity(pos_prefix, neg_prefix, bins=min(args.buckets, codes.shape[1] + 1)))
        print_histogram('SID hamming positive-rate bucket', bucket_by_similarity(-pos_hamming, -neg_hamming, bins=min(args.buckets, codes.shape[1] + 1)))
        retrieval_rows = retrieval_metrics_from_scorer(
            lambda anchor: sid_anchor_similarity(codes, anchor),
            len(item_ids),
            relevance,
            topks=topks,
            max_anchors=args.max_anchors,
            rng=rng,
        )
        print_retrieval_report(f'sid:{args.sid_coder}/{args.sid_export}', retrieval_rows)
        reports['spaces'][f'sid:{args.sid_coder}/{args.sid_export}:retrieval'] = retrieval_rows

    if score_rows:
        print_score_report(score_rows)
    else:
        print_section('no representation spaces requested')
        print('  pass --embedding-model and/or --sid-coder')

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + '\n')
        print()
        print(f'wrote summary json to {output_path}')


if __name__ == '__main__':
    main()
