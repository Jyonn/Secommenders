import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def fmt_float(value):
    if value is None:
        return '-'
    if isinstance(value, float) and math.isnan(value):
        return '-'
    return f'{float(value):.6g}'


def fmt_int(value):
    return f'{int(value):,}'


def print_table(headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    widths = []
    for index, header in enumerate(headers):
        widths.append(max([len(str(header)), *[len(row[index]) for row in rows]], default=len(str(header))))
    print('  '.join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        print('  '.join(row[index].ljust(widths[index]) for index in range(len(headers))))


def l2_normalize(matrix: np.ndarray):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (matrix / norms).astype(np.float32)


def load_item_ids(path: Path):
    frame = pd.read_parquet(path)
    column = 'pid' if 'pid' in frame.columns else frame.columns[0]
    return [str(item_id) for item_id in frame[column].tolist()]


def stable_sample(values: Iterable[str], limit: int | None, seed: int):
    values = sorted(str(value) for value in values)
    if limit is None or limit <= 0 or len(values) <= limit:
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=limit, replace=False))
    return [values[int(index)] for index in indices]


def describe(values):
    if not values:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'p50': None,
            'p90': None,
            'min': None,
            'max': None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.shape[0]),
        'mean': float(array.mean()),
        'std': float(array.std(ddof=0)),
        'p50': float(np.quantile(array, 0.5)),
        'p90': float(np.quantile(array, 0.9)),
        'min': float(array.min()),
        'max': float(array.max()),
    }


@dataclass
class VectorArtifact:
    source: str
    data: str
    path: Path
    item_ids: list[str]
    embeddings: np.ndarray
    meta: dict

    @property
    def item_set(self):
        return set(self.item_ids)

    @property
    def index(self):
        return {item_id: index for index, item_id in enumerate(self.item_ids)}


def _clustered_meta_matches(meta: dict, args):
    embedding = meta.get('embedding') or {}
    if args.cluster_embedding_source and str(embedding.get('source') or '').lower() != args.cluster_embedding_source:
        return False
    if args.cluster_content_model and str(embedding.get('content_model') or '').lower() != args.cluster_content_model:
        return False
    if args.cluster_levels:
        resolved = ','.join(str(value) for value in meta.get('resolved_levels') or [])
        if resolved != str(args.cluster_levels):
            return False
    return True


def find_clustered_artifact(root: Path, data: str, args):
    if args.artifact_dir:
        artifact_dir = Path(str(args.artifact_dir).format(data=data))
        if not artifact_dir.is_absolute():
            artifact_dir = root / artifact_dir
        return artifact_dir

    base = root / 'artifacts' / 'clustered' / data
    candidates = []
    for meta_path in sorted(base.glob('*/meta.json')):
        meta = read_json_if_exists(meta_path) or {}
        artifact_dir = meta_path.parent
        if not _clustered_meta_matches(meta, args):
            continue
        if not (artifact_dir / 'item_embeddings.npy').exists() or not (artifact_dir / 'item_ids.parquet').exists():
            continue
        candidates.append(artifact_dir)
    if not candidates:
        raise FileNotFoundError(
            f'No clustered artifact found for data={data} under {base}. '
            'Check --cluster-embedding-source/--cluster-content-model/--cluster-levels.'
        )
    candidates.sort(key=lambda path: (path / 'meta.json').stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        print(f'warning: multiple clustered artifacts for {data}; using latest {candidates[0]}')
    return candidates[0]


def load_clustered_vectors(root: Path, data: str, args):
    artifact_dir = find_clustered_artifact(root, data, args)
    embeddings = np.load(artifact_dir / 'item_embeddings.npy').astype(np.float32)
    item_ids = load_item_ids(artifact_dir / 'item_ids.parquet')
    if len(item_ids) != int(embeddings.shape[0]):
        raise ValueError(f'clustered artifact item count mismatch: {artifact_dir}')
    return VectorArtifact(
        source='clustered',
        data=data,
        path=artifact_dir,
        item_ids=item_ids,
        embeddings=l2_normalize(embeddings),
        meta=read_json_if_exists(artifact_dir / 'meta.json') or {},
    )


def _trained_checkpoint_rows(root: Path, data: str, args):
    from searcher import search

    filters = {'data': data}
    for attr, key in [
        ('trained_model', 'model'),
        ('trained_repr_type', 'repr_type'),
        ('trained_task_type', 'task_type'),
        ('trained_repr_source_model', 'repr_source_model'),
        ('trained_seed', 'seed'),
    ]:
        value = getattr(args, attr)
        if value is not None:
            filters[key] = value
    rows = search(root, filters, all_seeds=True, limit=None)['runs']
    rows = [
        row for row in rows
        if row.get('checkpoint') == 'yes'
        and str(row.get('phase') or '').lower() == 'train'
        and str(row.get('status') or '').lower() in {'finished', 'completed', 'test_only_finished'}
    ]
    rows.sort(key=lambda row: (str(row.get('seed')), str(row.get('signature')), str(row.get('path'))))
    return rows


def _find_state_tensor(state: dict, wanted: str):
    candidates = {
        'uid_embedding': ['uid_embedding.weight'],
        'uid_head': ['uid_head.weight'],
    }.get(wanted, [wanted])
    if wanted == 'auto':
        candidates = ['uid_embedding.weight', 'uid_head.weight']
    for suffix in candidates:
        for key, value in state.items():
            if key == suffix or key.endswith('.' + suffix):
                return key, value
    return None, None


def _compiled_item_ids_from_run(run_dir: Path, meta: dict):
    compiled_dir = meta.get('compiled_dir')
    if not compiled_dir:
        raise ValueError(f'trained run has no compiled_dir in meta: {run_dir}')
    compiled_path = Path(compiled_dir)
    if not compiled_path.is_absolute():
        compiled_path = ROOT / compiled_path
    uid_path = compiled_path / 'vocab' / 'uid.json'
    payload = read_json_if_exists(uid_path)
    if not payload or 'raw_item_ids' not in payload:
        raise FileNotFoundError(f'compiled uid vocabulary not found: {uid_path}')
    return [str(item_id) for item_id in payload['raw_item_ids']]


def load_trained_vectors(root: Path, data: str, args):
    rows = _trained_checkpoint_rows(root, data, args)
    if not rows:
        raise FileNotFoundError(f'No trained checkpoint found for data={data} with requested filters.')
    if len(rows) > 1:
        print(f'warning: multiple trained checkpoints for {data}; using first match {rows[0]["path"]}')
    run_dir = Path(rows[0]['path'])
    meta = read_json_if_exists(run_dir / 'meta.json') or {}
    item_ids = _compiled_item_ids_from_run(run_dir, meta)

    import torch

    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu')
    state = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f'checkpoint does not contain a model state dict: {run_dir / "best.pt"}')
    tensor_key, tensor = _find_state_tensor(state, args.trained_vector)
    if tensor is None:
        raise ValueError(
            f'Unable to find trained item vector {args.trained_vector!r} in checkpoint {run_dir / "best.pt"}. '
            'Try --trained-vector uid_embedding or --trained-vector uid_head.'
        )
    embeddings = tensor.detach().cpu().numpy().astype(np.float32)
    if len(item_ids) != int(embeddings.shape[0]):
        raise ValueError(
            f'trained vector shape mismatch for {run_dir}: item_ids={len(item_ids)} '
            f'embeddings={embeddings.shape[0]} key={tensor_key}'
        )
    meta = dict(meta)
    meta['vector_key'] = tensor_key
    return VectorArtifact(
        source='trained',
        data=data,
        path=run_dir,
        item_ids=item_ids,
        embeddings=l2_normalize(embeddings),
        meta=meta,
    )


def load_vectors(root: Path, data: str, args):
    if args.source == 'clustered':
        return load_clustered_vectors(root, data, args)
    if args.source == 'trained':
        return load_trained_vectors(root, data, args)
    raise ValueError(f'unsupported vector source: {args.source}')


def exact_topk_neighbors(
    artifact: VectorArtifact,
    *,
    anchor_ids: list[str],
    candidate_ids: list[str],
    topk: int,
    anchor_chunk_size: int,
    candidate_chunk_size: int,
):
    index = artifact.index
    missing_anchors = [item_id for item_id in anchor_ids if item_id not in index]
    missing_candidates = [item_id for item_id in candidate_ids if item_id not in index]
    if missing_anchors:
        raise ValueError(f'{artifact.data} misses {len(missing_anchors)} anchor ids; first={missing_anchors[:5]}')
    if missing_candidates:
        raise ValueError(f'{artifact.data} misses {len(missing_candidates)} candidate ids; first={missing_candidates[:5]}')

    candidate_indices = np.asarray([index[item_id] for item_id in candidate_ids], dtype=np.int64)
    anchor_indices = np.asarray([index[item_id] for item_id in anchor_ids], dtype=np.int64)
    topn = min(int(topk), max(0, len(candidate_indices) - 1))
    if topn <= 0:
        return {item_id: [] for item_id in anchor_ids}

    neighbors = {}
    matrix = artifact.embeddings
    candidate_lookup = {int(row_index): position for position, row_index in enumerate(candidate_indices.tolist())}

    for anchor_start in range(0, len(anchor_indices), anchor_chunk_size):
        anchor_end = min(anchor_start + anchor_chunk_size, len(anchor_indices))
        anchor_block = anchor_indices[anchor_start:anchor_end]
        anchor_vectors = matrix[anchor_block]
        best_scores = np.full((len(anchor_block), topn), -np.inf, dtype=np.float32)
        best_indices = np.full((len(anchor_block), topn), -1, dtype=np.int64)

        for candidate_start in range(0, len(candidate_indices), candidate_chunk_size):
            candidate_end = min(candidate_start + candidate_chunk_size, len(candidate_indices))
            candidate_block = candidate_indices[candidate_start:candidate_end]
            scores = anchor_vectors @ matrix[candidate_block].T
            for local_anchor, anchor_row in enumerate(anchor_block.tolist()):
                candidate_position = candidate_lookup.get(int(anchor_row))
                if candidate_position is not None and candidate_start <= candidate_position < candidate_end:
                    scores[local_anchor, candidate_position - candidate_start] = -np.inf

            local_topn = min(topn, scores.shape[1])
            local_part = np.argpartition(-scores, local_topn - 1, axis=1)[:, :local_topn]
            local_scores = np.take_along_axis(scores, local_part, axis=1)
            local_indices = candidate_block[local_part]

            merged_scores = np.concatenate([best_scores, local_scores], axis=1)
            merged_indices = np.concatenate([best_indices, local_indices], axis=1)
            merged_part = np.argpartition(-merged_scores, topn - 1, axis=1)[:, :topn]
            best_scores = np.take_along_axis(merged_scores, merged_part, axis=1)
            best_indices = np.take_along_axis(merged_indices, merged_part, axis=1)
            order = np.argsort(-best_scores, axis=1)
            best_scores = np.take_along_axis(best_scores, order, axis=1)
            best_indices = np.take_along_axis(best_indices, order, axis=1)

        for local_anchor, anchor_item_id in enumerate(anchor_ids[anchor_start:anchor_end]):
            neighbors[anchor_item_id] = [
                artifact.item_ids[int(row_index)]
                for row_index in best_indices[local_anchor].tolist()
                if int(row_index) >= 0
            ]
    return neighbors


def overlap_scores(left_neighbors: dict[str, list[str]], right_neighbors: dict[str, list[str]], topk: int):
    scores = []
    for item_id, left in left_neighbors.items():
        right = right_neighbors.get(item_id)
        if right is None:
            continue
        scores.append(len(set(left) & set(right)) / float(topk))
    return scores


def retention_scores(full_reference_neighbors: dict[str, list[str]], candidate_set: set[str], topk: int):
    scores = []
    for neighbors in full_reference_neighbors.values():
        scores.append(len(set(neighbors) & candidate_set) / float(topk))
    return scores


def evaluate_scale(reference: VectorArtifact, target: VectorArtifact, core_items: set[str], args):
    ref_set = reference.item_set
    target_set = target.item_set
    conditional_items = ref_set & target_set
    if len(conditional_items) <= args.topk:
        raise ValueError(f'Not enough shared items for {target.data}: shared={len(conditional_items)} topk={args.topk}')

    conditional_anchors = stable_sample(conditional_items, args.max_anchors, args.seed)
    core_anchors = stable_sample(core_items, args.max_anchors, args.seed)
    conditional_candidates = sorted(conditional_items)
    core_candidates = sorted(core_items)
    ref_full_candidates = sorted(ref_set)

    print(
        f'evaluating {target.data}: shared={len(conditional_items):,} core={len(core_items):,} '
        f'anchors={len(conditional_anchors):,}/{len(core_anchors):,}'
    )

    ref_conditional = exact_topk_neighbors(
        reference,
        anchor_ids=conditional_anchors,
        candidate_ids=conditional_candidates,
        topk=args.topk,
        anchor_chunk_size=args.anchor_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    target_conditional = exact_topk_neighbors(
        target,
        anchor_ids=conditional_anchors,
        candidate_ids=conditional_candidates,
        topk=args.topk,
        anchor_chunk_size=args.anchor_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    conditional_scores = overlap_scores(ref_conditional, target_conditional, args.topk)

    ref_full = exact_topk_neighbors(
        reference,
        anchor_ids=conditional_anchors,
        candidate_ids=ref_full_candidates,
        topk=args.topk,
        anchor_chunk_size=args.anchor_chunk_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    retention = retention_scores(ref_full, target_set, args.topk)

    core_scores = []
    if len(core_candidates) > args.topk:
        ref_core = exact_topk_neighbors(
            reference,
            anchor_ids=core_anchors,
            candidate_ids=core_candidates,
            topk=args.topk,
            anchor_chunk_size=args.anchor_chunk_size,
            candidate_chunk_size=args.candidate_chunk_size,
        )
        target_core = exact_topk_neighbors(
            target,
            anchor_ids=core_anchors,
            candidate_ids=core_candidates,
            topk=args.topk,
            anchor_chunk_size=args.anchor_chunk_size,
            candidate_chunk_size=args.candidate_chunk_size,
        )
        core_scores = overlap_scores(ref_core, target_core, args.topk)

    return {
        'data': target.data,
        'source': target.source,
        'artifact_path': str(target.path),
        'item_count': len(target.item_ids),
        'reference_item_count': len(reference.item_ids),
        'coverage': len(target_set & ref_set) / float(len(ref_set)),
        'conditional_item_count': len(conditional_items),
        'core_item_count': len(core_items),
        'conditional_anchor_count': len(conditional_anchors),
        'core_anchor_count': len(core_anchors),
        'conditional_stability': describe(conditional_scores),
        'core_stability': describe(core_scores),
        'full_neighbor_retention': describe(retention),
    }


def run_neighbor_stability(args):
    root = Path(args.root).resolve()
    datasets = [part.strip().lower() for part in args.data.split(',') if part.strip()]
    if not datasets:
        raise ValueError('--data must contain at least one dataset')
    reference_data = args.reference_data.lower()

    print(f'loading reference {reference_data} source={args.source}')
    reference = load_vectors(root, reference_data, args)
    targets = []
    for data in datasets:
        print(f'loading target {data} source={args.source}')
        targets.append(load_vectors(root, data, args))

    core_items = set(reference.item_ids)
    for artifact in targets:
        core_items &= artifact.item_set
    if len(core_items) <= args.topk:
        raise ValueError(f'core item set too small: core={len(core_items)} topk={args.topk}')

    report = {
        'task': 'neighbor_stability',
        'source': args.source,
        'reference_data': reference_data,
        'datasets': datasets,
        'topk': args.topk,
        'max_anchors': args.max_anchors,
        'seed': args.seed,
        'reference': {
            'data': reference.data,
            'path': str(reference.path),
            'item_count': len(reference.item_ids),
            'meta': reference.meta,
        },
        'core_item_count': len(core_items),
        'results': [],
    }

    for target in targets:
        report['results'].append(evaluate_scale(reference, target, core_items, args))

    rows = []
    for result in report['results']:
        rows.append(
            [
                result['data'],
                fmt_int(result['item_count']),
                fmt_float(result['coverage']),
                fmt_int(result['conditional_item_count']),
                fmt_int(result['conditional_anchor_count']),
                fmt_float(result['conditional_stability']['mean']),
                fmt_float(result['core_stability']['mean']),
                fmt_float(result['full_neighbor_retention']['mean']),
            ]
        )
    print()
    print_table(
        [
            'data',
            'items',
            'coverage',
            'shared',
            'anchors',
            f'conditional@{args.topk}',
            f'core@{args.topk}',
            f'retention@{args.topk}',
        ],
        rows,
    )

    if args.output:
        write_json(Path(args.output), report)
        print(f'wrote neighbor stability report to {args.output}')
    return report


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluation utilities for Secommenders artifacts.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    stability = subparsers.add_parser(
        'neighbor-stability',
        help='Measure Top-K item-neighbor stability against a reference scale.',
        description=(
            'Compare item-neighbor stability between a reference dataset, usually ra99/rv99, '
            'and smaller scales. The evaluator reports conditional stability on each shared '
            'vocabulary, core stability on the vocabulary shared by all scales, and full-neighbor '
            'retention to separate vocabulary coverage from embedding drift.'
        ),
    )
    stability.add_argument('--root', default=str(ROOT), help='Algorithm repository root.')
    stability.add_argument('--source', choices=['clustered', 'trained'], default='clustered')
    stability.add_argument('--reference-data', default='ra99')
    stability.add_argument('--data', required=True, help='Comma-separated target datasets, e.g. ra1,ra2,ra5,ra10.')
    stability.add_argument('--topk', type=int, default=20)
    stability.add_argument('--max-anchors', type=int, default=5000, help='Stable sampled anchors; <=0 means all anchors.')
    stability.add_argument('--seed', type=int, default=20260717)
    stability.add_argument('--anchor-chunk-size', type=int, default=64)
    stability.add_argument('--candidate-chunk-size', type=int, default=100000)
    stability.add_argument('--output', default=None, help='Optional JSON report path.')

    stability.add_argument('--artifact-dir', default=None, help='Explicit vector artifact dir; may include {data}.')
    stability.add_argument('--cluster-embedding-source', default='collaborative', choices=['collaborative', 'content', 'concat'])
    stability.add_argument('--cluster-content-model', default=None)
    stability.add_argument('--cluster-levels', default=None, help='Resolved levels string, e.g. 132 or 20,20.')

    stability.add_argument('--trained-model', default=None)
    stability.add_argument('--trained-repr-type', default=None)
    stability.add_argument('--trained-task-type', default=None)
    stability.add_argument('--trained-repr-source-model', default=None)
    stability.add_argument('--trained-seed', type=int, default=None)
    stability.add_argument('--trained-vector', default='auto', choices=['auto', 'uid_embedding', 'uid_head'])
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.max_anchors is not None and args.max_anchors <= 0:
        args.max_anchors = None
    if args.command == 'neighbor-stability':
        run_neighbor_stability(args)
    else:
        raise ValueError(f'unsupported command: {args.command}')


if __name__ == '__main__':
    main()
