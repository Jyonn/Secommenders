import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pigmento import pnt

from utils.artifact import ArtifactStore
from utils.compile import normalize_model_name
from utils.logging import setup_logging


EPS = 1e-12


@dataclass
class CompareTarget:
    method: str
    export: str

    @property
    def label(self):
        return f'{self.method}.{self.export}'


@dataclass
class LoadedExport:
    target: CompareTarget
    export_dir: Path
    item_ids: list[str]
    item_index: dict[str, int]
    codes: np.ndarray
    quantized: np.ndarray
    meta: dict


METRIC_SPECS = {
    'pair_cos_pearson': dict(title='Pair Cos Pearson ↑', better='max', kind='float4'),
    'pair_cos_spearman': dict(title='Pair Cos Spearman ↑', better='max', kind='float4'),
    'subset_knn_recall@20': dict(title='Subset KNN R@20 ↑', better='max', kind='pct2'),
    'subset_knn_recall@50': dict(title='Subset KNN R@50 ↑', better='max', kind='pct2'),
    'rmse': dict(title='RMSE ↓', better='min', kind='float4'),
    'cosine_mean': dict(title='Mean Cosine ↑', better='max', kind='float4'),
    'relative_l2': dict(title='Rel L2 ↓', better='min', kind='float4'),
    'unique_code_ratio': dict(title='Unique Code Ratio ↑', better='max', kind='pct2'),
    'collision_rate': dict(title='Collision Rate ↓', better='min', kind='pct2'),
    'collided_item_ratio': dict(title='Collided Item Ratio ↓', better='min', kind='pct2'),
    'max_collision_size': dict(title='Max Collision ↓', better='min', kind='int'),
    'slot_entropy_mean': dict(title='Slot Entropy Mean ↑', better='max', kind='float4'),
    'dead_code_ratio': dict(title='Dead Code Ratio ↓', better='min', kind='pct2'),
}


def parse_compare(compare: str):
    targets = []
    seen = set()
    for chunk in str(compare).split('+'):
        token = chunk.strip()
        if not token:
            continue
        if '.' not in token:
            raise ValueError(f'Invalid compare token "{token}". Expected <method>.<export>.')
        method, export = token.rsplit('.', 1)
        target = CompareTarget(method=method.strip().lower(), export=export.strip())
        if not target.method or not target.export:
            raise ValueError(f'Invalid compare token "{token}". Expected <method>.<export>.')
        if target.label in seen:
            raise ValueError(f'Duplicate compare target: {target.label}')
        seen.add(target.label)
        targets.append(target)
    if len(targets) < 2:
        raise ValueError('Please provide at least two compare targets, e.g. rqvae.loss+basic-rqvae.loss')
    return targets


def read_first_column_values(path: Path):
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    column = frame.columns[0]
    return frame[column].astype(str).tolist()


def ensure_2d(array: np.ndarray):
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D array, got shape {list(array.shape)}')
    return array


def load_source_embeddings(store: ArtifactStore, model: str):
    embedding_dir = store.embedded_dir(model)
    embedding_path = embedding_dir / 'embeddings.npy'
    item_ids_path = embedding_dir / 'item_ids.parquet'
    if not embedding_path.exists():
        raise FileNotFoundError(f'Source embeddings not found: {embedding_path}')
    if not item_ids_path.exists():
        raise FileNotFoundError(f'Source embedding item ids not found: {item_ids_path}')
    pnt(f'loading source embeddings from {embedding_path}')
    embeddings = np.load(embedding_path).astype(np.float32)
    item_ids = read_first_column_values(item_ids_path)
    if len(item_ids) != len(embeddings):
        raise ValueError(
            f'Source item id count {len(item_ids)} does not match embedding rows {len(embeddings)}'
        )
    return embedding_dir, embeddings, item_ids


def load_export(store: ArtifactStore, model: str, target: CompareTarget):
    export_dir = store.quantized_dir(model, target.method) / 'exports' / target.export
    codes_path = export_dir / 'codebook_indices.npy'
    quantized_path = export_dir / 'quantized_latents.npy'
    item_ids_path = export_dir / 'item_ids.parquet'
    meta_path = export_dir / 'meta.json'
    missing = [path for path in [codes_path, quantized_path, item_ids_path, meta_path] if not path.exists()]
    if missing:
        missing_text = ', '.join(str(path) for path in missing)
        raise FileNotFoundError(f'Export {target.label} is incomplete under {export_dir}: {missing_text}')

    pnt(f'loading export {target.label} from {export_dir}')
    codes = ensure_2d(np.load(codes_path))
    quantized = ensure_2d(np.load(quantized_path).astype(np.float32))
    item_ids = read_first_column_values(item_ids_path)
    meta = json.loads(meta_path.read_text())

    if len(item_ids) != len(codes):
        raise ValueError(
            f'Export {target.label} item id count {len(item_ids)} does not match code rows {len(codes)}'
        )
    if len(item_ids) != len(quantized):
        raise ValueError(
            f'Export {target.label} item id count {len(item_ids)} does not match quantized rows {len(quantized)}'
        )
    return LoadedExport(
        target=target,
        export_dir=export_dir,
        item_ids=item_ids,
        item_index={item_id: index for index, item_id in enumerate(item_ids)},
        codes=codes,
        quantized=quantized,
        meta=meta,
    )


def l2_normalize(matrix: np.ndarray):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, EPS, None)


def sample_pairs(num_items: int, num_pairs: int, rng: np.random.Generator):
    if num_items < 2:
        raise ValueError('Need at least two items to sample pair similarities')
    num_pairs = min(int(num_pairs), num_items * (num_items - 1))
    left = rng.integers(0, num_items, size=num_pairs, dtype=np.int64)
    right = rng.integers(0, num_items, size=num_pairs, dtype=np.int64)
    same_mask = left == right
    while same_mask.any():
        right[same_mask] = rng.integers(0, num_items, size=int(same_mask.sum()), dtype=np.int64)
        same_mask = left == right
    return left, right


def corrcoef_safe(left: np.ndarray, right: np.ndarray):
    if left.size < 2 or right.size < 2:
        return float('nan')
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def rank_array(values: np.ndarray):
    return pd.Series(values).rank(method='average').to_numpy(dtype=np.float64)


def topk_indices(similarity: np.ndarray, k: int):
    if k <= 0:
        return np.empty((similarity.shape[0], 0), dtype=np.int64)
    work = similarity.copy()
    np.fill_diagonal(work, -np.inf)
    indices = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    row_ids = np.arange(work.shape[0])[:, None]
    scores = work[row_ids, indices]
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(indices, order, axis=1)


def subset_knn_recall(anchor_norm: np.ndarray, candidate_norm: np.ndarray, ks: list[int]):
    similarity_anchor = anchor_norm @ anchor_norm.T
    similarity_candidate = candidate_norm @ candidate_norm.T
    metrics = {}
    for k in ks:
        effective_k = min(k, anchor_norm.shape[0] - 1)
        if effective_k <= 0:
            metrics[f'subset_knn_recall@{k}'] = float('nan')
            continue
        top_anchor = topk_indices(similarity_anchor, effective_k)
        top_candidate = topk_indices(similarity_candidate, effective_k)
        recall_values = []
        for anchor_row, candidate_row in zip(top_anchor, top_candidate):
            recall_values.append(len(set(anchor_row.tolist()) & set(candidate_row.tolist())) / effective_k)
        metrics[f'subset_knn_recall@{k}'] = float(np.mean(recall_values))
    return metrics


def resolve_slot_sizes(meta: dict, codes: np.ndarray):
    quantizer_config = meta.get('quantizer_config') or {}
    num_emb_list = quantizer_config.get('num_emb_list')
    if isinstance(num_emb_list, list) and len(num_emb_list) == codes.shape[1]:
        return [max(int(size), int(codes[:, index].max()) + 1) for index, size in enumerate(num_emb_list)]

    codebook_size = quantizer_config.get('codebook_size')
    if codebook_size is not None:
        size = int(codebook_size)
        return [max(size, int(codes[:, index].max()) + 1) for index in range(codes.shape[1])]

    return [int(codes[:, index].max()) + 1 for index in range(codes.shape[1])]


def compute_code_metrics(codes: np.ndarray, meta: dict):
    unique_rows, counts = np.unique(codes, axis=0, return_counts=True)
    num_items = len(codes)
    collision_counts = counts[counts > 1]
    collision_group_count = int(len(collision_counts))
    collided_item_count = int(collision_counts.sum()) if collision_group_count else 0
    max_collision_size = int(collision_counts.max()) if collision_group_count else 1

    slot_sizes = resolve_slot_sizes(meta, codes)
    entropies = []
    dead_ratios = []
    for index, slot_size in enumerate(slot_sizes):
        slot = codes[:, index].astype(np.int64)
        bincount = np.bincount(slot, minlength=int(slot_size)).astype(np.float64)
        used = int(np.count_nonzero(bincount))
        probabilities = bincount[bincount > 0] / max(float(num_items), 1.0)
        entropy = 0.0
        if probabilities.size and slot_size > 1:
            entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(slot_size))
        entropies.append(entropy)
        dead_ratios.append(float(1.0 - (used / max(slot_size, 1))))

    return {
        'unique_code_ratio': float(len(unique_rows) / max(num_items, 1)),
        'collision_rate': float(1.0 - (len(unique_rows) / max(num_items, 1))),
        'collided_item_ratio': float(collided_item_count / max(num_items, 1)),
        'max_collision_size': max_collision_size,
        'slot_entropy_mean': float(np.mean(entropies)) if entropies else float('nan'),
        'dead_code_ratio': float(np.mean(dead_ratios)) if dead_ratios else float('nan'),
    }


def compute_direct_metrics(source_embeddings: np.ndarray, quantized: np.ndarray):
    if source_embeddings.shape[1] != quantized.shape[1]:
        return {
            'rmse': float('nan'),
            'cosine_mean': float('nan'),
            'relative_l2': float('nan'),
        }
    delta = quantized - source_embeddings
    source_norm = np.linalg.norm(source_embeddings, axis=1)
    quantized_norm = l2_normalize(quantized)
    source_unit = l2_normalize(source_embeddings)
    relative_l2 = np.linalg.norm(delta, axis=1) / np.clip(source_norm, EPS, None)
    cosine_mean = np.sum(source_unit * quantized_norm, axis=1).mean()
    return {
        'rmse': float(np.sqrt(np.mean(np.square(delta)))),
        'cosine_mean': float(cosine_mean),
        'relative_l2': float(np.mean(relative_l2)),
    }


def format_value(value, kind: str):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 'N/A'
    if kind == 'int':
        return str(int(round(value)))
    if kind == 'pct2':
        return f'{value * 100:.2f}%'
    if kind == 'float4':
        return f'{value:.4f}'
    return str(value)


def is_better(value: float, best_value: float, better: str):
    if math.isnan(value) or math.isnan(best_value):
        return False
    tolerance = 1e-12
    if better == 'max':
        return value >= best_value - tolerance
    return value <= best_value + tolerance


def render_table(title: str, rows: list[dict], metric_names: list[str]):
    if not rows:
        return
    print()
    print(title)
    headers = ['Method'] + [METRIC_SPECS[name]['title'] for name in metric_names]
    best_values = {}
    for metric_name in metric_names:
        values = [
            row[metric_name] for row in rows
            if isinstance(row[metric_name], (int, float)) and not math.isnan(row[metric_name])
        ]
        best_values[metric_name] = (
            max(values) if values and METRIC_SPECS[metric_name]['better'] == 'max'
            else min(values) if values
            else float('nan')
        )

    body = []
    for row in rows:
        rendered = [row['method']]
        for metric_name in metric_names:
            value = row[metric_name]
            cell = format_value(value, METRIC_SPECS[metric_name]['kind'])
            if cell != 'N/A' and is_better(float(value), float(best_values[metric_name]), METRIC_SPECS[metric_name]['better']):
                cell = f'{cell} *'
            rendered.append(cell)
        body.append(rendered)

    widths = []
    for column_index in range(len(headers)):
        widths.append(max(len(headers[column_index]), *(len(row[column_index]) for row in body)))

    def render_line(values):
        return '  '.join(value.ljust(widths[index]) for index, value in enumerate(values))

    print(render_line(headers))
    print(render_line(['-' * width for width in widths]))
    for row in body:
        print(render_line(row))
    print('* = best in column')


def render_method_manifest(rows: list[dict]):
    print()
    print('Methods')
    headers = ['Method', 'Items', 'Codes', 'Quantized Dim', 'Direct Recon', 'Export Dir']
    body = []
    for row in rows:
        body.append([
            row['method'],
            str(row['item_count']),
            f"{row['num_slots']} x {row['slot_size_hint']}",
            str(row['quantized_dim']),
            'yes' if row['direct_recon_available'] else 'no',
            row['export_dir'],
        ])

    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

    def render_line(values):
        return '  '.join(value.ljust(widths[index]) for index, value in enumerate(values))

    print(render_line(headers))
    print(render_line(['-' * width for width in widths]))
    for row in body:
        print(render_line(row))


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description='Compare multiple quantized SID exports with user-friendly terminal output.')
    parser.add_argument('--data', required=True, help='Dataset name, such as mind.')
    parser.add_argument('--model', required=True, help='Source embedding model name, such as llama3.')
    parser.add_argument('--compare', required=True, help='Compare targets like rqvae.loss+basic-rqvae.loss+opqvae.recon')
    parser.add_argument('--pair_samples', type=int, default=100000, help='Number of random item pairs for similarity correlation.')
    parser.add_argument('--subset_items', type=int, default=1024, help='Subset size for subset-KNN recall.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subset and pair sampling.')
    args = parser.parse_args()

    data = str(args.data).lower()
    model = normalize_model_name(args.model)
    targets = parse_compare(args.compare)
    store = ArtifactStore(data)

    _, source_embeddings, source_item_ids = load_source_embeddings(store, model)
    source_index = {item_id: index for index, item_id in enumerate(source_item_ids)}

    exports = [load_export(store, model, target) for target in targets]
    common_item_set = set(source_item_ids)
    for loaded in exports:
        common_item_set &= set(loaded.item_ids)
    common_item_ids = [item_id for item_id in source_item_ids if item_id in common_item_set]
    if not common_item_ids:
        raise ValueError('No common items found across source embeddings and all compare targets')

    pnt(
        f'common items across source embeddings and {len(exports)} exports: '
        f'{len(common_item_ids)}/{len(source_item_ids)}'
    )

    common_source_indices = np.asarray([source_index[item_id] for item_id in common_item_ids], dtype=np.int64)
    source_common = source_embeddings[common_source_indices]
    source_norm_common = l2_normalize(source_common)

    rng = np.random.default_rng(int(args.seed))
    subset_size = min(int(args.subset_items), len(common_item_ids))
    subset_indices = np.sort(rng.choice(len(common_item_ids), size=subset_size, replace=False))
    pair_left, pair_right = sample_pairs(len(common_item_ids), int(args.pair_samples), rng)

    source_subset_norm = source_norm_common[subset_indices]
    source_pair_cos = np.sum(source_norm_common[pair_left] * source_norm_common[pair_right], axis=1)

    rows = []
    for loaded in exports:
        common_export_indices = np.asarray([loaded.item_index[item_id] for item_id in common_item_ids], dtype=np.int64)
        quantized_common = loaded.quantized[common_export_indices]
        quantized_norm_common = l2_normalize(quantized_common)

        pair_quantized_cos = np.sum(
            quantized_norm_common[pair_left] * quantized_norm_common[pair_right],
            axis=1,
        )
        subset_quantized_norm = quantized_norm_common[subset_indices]

        metrics = {}
        metrics.update(compute_direct_metrics(source_common, quantized_common))
        metrics.update(compute_code_metrics(loaded.codes[common_export_indices], loaded.meta))
        metrics['pair_cos_pearson'] = corrcoef_safe(source_pair_cos, pair_quantized_cos)
        metrics['pair_cos_spearman'] = corrcoef_safe(rank_array(source_pair_cos), rank_array(pair_quantized_cos))
        metrics.update(subset_knn_recall(source_subset_norm, subset_quantized_norm, ks=[20, 50]))

        slot_sizes = resolve_slot_sizes(loaded.meta, loaded.codes)
        slot_size_hint = slot_sizes[0] if slot_sizes and len(set(slot_sizes)) == 1 else 'mixed'

        row = {
            'method': loaded.target.label,
            'export_dir': str(loaded.export_dir),
            'item_count': len(common_item_ids),
            'num_slots': int(loaded.codes.shape[1]),
            'slot_size_hint': slot_size_hint,
            'quantized_dim': int(loaded.quantized.shape[1]),
            'direct_recon_available': bool(source_common.shape[1] == loaded.quantized.shape[1]),
        }
        row.update(metrics)
        rows.append(row)

    print()
    print(f'Compare SID Exports for data={data} model={model}')
    print(f'Common items: {len(common_item_ids)} / {len(source_item_ids)}')
    print(f'Pair samples: {len(pair_left)}')
    print(f'Subset items for KNN: {subset_size}')
    print('Notes: direct reconstruction metrics are only available when quantized_dim == source embedding dim.')

    render_method_manifest(rows)
    render_table(
        'Geometry and Structure',
        rows,
        ['pair_cos_pearson', 'pair_cos_spearman', 'subset_knn_recall@20', 'subset_knn_recall@50'],
    )
    render_table(
        'Direct Reconstruction',
        rows,
        ['rmse', 'cosine_mean', 'relative_l2'],
    )
    render_table(
        'Code Quality',
        rows,
        ['unique_code_ratio', 'collision_rate', 'collided_item_ratio', 'max_collision_size', 'slot_entropy_mean', 'dead_code_ratio'],
    )


if __name__ == '__main__':
    main()
