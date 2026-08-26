import argparse
import json
import math
from pathlib import Path


def _load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _representation_graph(config):
    graph = config.get('representation_graph')
    if isinstance(graph, dict):
        return graph
    meta = config.get('_meta')
    if isinstance(meta, dict) and isinstance(meta.get('representation_graph'), dict):
        return meta['representation_graph']
    return None


def _representation_names(config, matrix_size):
    graph = _representation_graph(config)
    if graph:
        encoder = graph.get('encoder') or {}
        names = encoder.get('representations') or []
        if len(names) + 1 == matrix_size:
            return ['model', *names], graph.get('representations') or {}

    repr_type = str(config.get('repr_type') or '').strip()
    names = [part for part in repr_type.split('+') if part]
    if len(names) + 1 == matrix_size:
        return ['model', *names], {}
    return [f'repr_{index}' for index in range(matrix_size)], {}


def _kind_for(name, catalog):
    if name == 'model':
        return 'model'
    spec = catalog.get(name) if isinstance(catalog, dict) else None
    if isinstance(spec, dict) and spec.get('type'):
        return str(spec['type']).lower()
    lowered = name.lower()
    if lowered.startswith('sid'):
        return 'sid'
    if lowered.startswith('uid'):
        return 'uid'
    if lowered.startswith(('embedding', 'emb')):
        return 'embedding'
    return lowered


def _display_labels(names, catalog):
    kinds = [_kind_for(name, catalog) for name in names]
    counts = {kind: kinds.count(kind) for kind in set(kinds)}
    short = {'model': 'Model', 'sid': 'SID', 'uid': 'UID', 'embedding': 'Emb', 'hash': 'Hash', 'text': 'Text'}
    labels = []
    for name, kind in zip(names, kinds):
        label = short.get(kind, kind.title())
        if counts[kind] > 1:
            label = f'{label}[{name}]'
        labels.append(label)
    return labels


def _format_matrix(title, matrix, labels, precision):
    number_width = max(9, precision + 6)
    label_width = max(7, max(len(label) for label in labels))
    corner_label = 'Query / Key'
    lines = [title, f'{corner_label:<{label_width}} ' + ' '.join(
        f'{label:>{number_width}}' for label in labels
    )]
    lines.append('-' * len(lines[-1]))
    for label, row in zip(labels, matrix):
        values = ' '.join(f'{value:>{number_width}.{precision}f}' for value in row)
        lines.append(f'{label:<{label_width}} {values}')
    return '\n'.join(lines)


def _analyze(matrix, labels):
    row_centered = []
    relations = []
    for query, row in zip(labels, matrix):
        mean = sum(row) / len(row)
        centered = [value - mean for value in row]
        row_centered.append(centered)
        strongest = max(range(len(row)), key=lambda index: centered[index])
        weakest = min(range(len(row)), key=lambda index: centered[index])
        spread = centered[strongest] - centered[weakest]
        relations.append({
            'query': query,
            'preferred_key': labels[strongest],
            'preferred_bias': centered[strongest],
            'suppressed_key': labels[weakest],
            'suppressed_bias': centered[weakest],
            'odds_ratio': math.exp(min(spread, 50.0)),
        })
    return row_centered, relations


def _load_checkpoint(path):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit('PyTorch is required to inspect a checkpoint.') from exc
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _find_bias(state_dict):
    matches = [
        (name, value) for name, value in state_dict.items()
        if name == 'representation_pair_bias' or name.endswith('.representation_pair_bias')
    ]
    if not matches:
        raise ValueError(
            'checkpoint does not contain representation_pair_bias; '
            'the run was probably trained with representation_pair_bias=false'
        )
    if len(matches) > 1:
        names = ', '.join(name for name, _ in matches)
        raise ValueError(f'checkpoint contains multiple pair-bias matrices: {names}')
    return matches[0]


def _selected_matrices(tensor, head):
    values = tensor.detach().float().cpu()
    if values.ndim == 2:
        return [('shared', values.tolist())], 'shared'
    if values.ndim != 3:
        raise ValueError(
            f'representation_pair_bias must have shape [R,R] or [H,R,R], got {list(values.shape)}'
        )
    head_count = int(values.shape[0])
    selection = str(head).strip().lower()
    if selection == 'mean':
        return [('head mean', values.mean(dim=0).tolist())], 'head'
    if selection == 'all':
        return [(f'head {index}', values[index].tolist()) for index in range(head_count)], 'head'
    try:
        index = int(selection)
    except ValueError as exc:
        raise ValueError('--head expects mean, all, or a zero-based head index') from exc
    if not 0 <= index < head_count:
        raise ValueError(f'head index {index} is outside [0, {head_count - 1}]')
    return [(f'head {index}', values[index].tolist())], 'head'


def inspect_checkpoint(path, precision=4, include_model=True, as_json=False, head='mean'):
    checkpoint = _load_checkpoint(path)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError('checkpoint does not contain a usable model_state_dict')
    parameter_name, tensor = _find_bias(state_dict)
    selected_matrices, mode = _selected_matrices(tensor, head)
    matrix_size = len(selected_matrices[0][1])
    if not matrix_size or any(
        len(row) != matrix_size
        for _, matrix in selected_matrices
        for row in matrix
    ):
        raise ValueError(f'{parameter_name} must contain non-empty square matrices')

    config = checkpoint.get('config') if isinstance(checkpoint.get('config'), dict) else {}
    if not config:
        config = _load_json(path.parent / 'meta.json').get('config') or {}
    names, catalog = _representation_names(config, matrix_size)
    labels = _display_labels(names, catalog)

    if not include_model and names and names[0] == 'model':
        names = names[1:]
        labels = labels[1:]
        selected_matrices = [
            (title, [row[1:] for row in matrix[1:]])
            for title, matrix in selected_matrices
        ]

    analyses = []
    for title, matrix in selected_matrices:
        centered, relations = _analyze(matrix, labels)
        analyses.append({
            'title': title,
            'matrix': matrix,
            'row_centered_matrix': centered,
            'relations': relations,
        })
    result = {
        'checkpoint': str(path),
        'epoch': checkpoint.get('epoch'),
        'parameter': parameter_name,
        'mode': mode,
        'parameter_shape': list(tensor.shape),
        'orientation': 'rows=query, columns=key',
        'representations': [
            {'name': name, 'label': label, 'kind': _kind_for(name, catalog)}
            for name, label in zip(names, labels)
        ],
        'analyses': analyses,
    }
    if as_json:
        print(json.dumps(result, indent=2))
        return result

    epoch = checkpoint.get('epoch')
    print(f'checkpoint : {path}')
    print(f'epoch      : {epoch if epoch is not None else "-"}')
    print(f'parameter  : {parameter_name}')
    print(f'mode       : {mode} shape={list(tensor.shape)}')
    print('orientation: rows are Query representations; columns are Key representations')
    for analysis in analyses:
        suffix = '' if analysis['title'] == 'shared' else f' ({analysis["title"]})'
        print()
        print(_format_matrix(
            f'-- raw attention-logit bias{suffix} --', analysis['matrix'], labels, precision
        ))
        print()
        print(_format_matrix(
            f'-- effective preference{suffix} (row-centered) --',
            analysis['row_centered_matrix'], labels, precision,
        ))
        print()
        print(f'-- strongest preference within each query row{suffix} --')
        for relation in analysis['relations']:
            print(
                f'  {relation["query"]:<18} prefers {relation["preferred_key"]:<18} '
                f'over {relation["suppressed_key"]:<18} '
                f'logit_gap={relation["preferred_bias"] - relation["suppressed_bias"]:.{precision}f} '
                f'attention_odds≈{relation["odds_ratio"]:.2f}x'
            )
    print()
    print('Note: row-centered values are more meaningful because adding the same constant')
    print('to every key in one query row does not change the attention softmax.')
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Inspect and explain a learned representation pair-bias matrix.',
    )
    parser.add_argument('checkpoint', type=Path, help='path to trainer best.pt')
    parser.add_argument('--precision', type=int, default=4)
    parser.add_argument('--exclude-model', action='store_true', help='show only item representations')
    parser.add_argument('--head', default='mean', help='per-head selection: mean, all, or zero-based index')
    parser.add_argument('--json', action='store_true', help='emit machine-readable JSON')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f'checkpoint not found: {args.checkpoint}')
    try:
        inspect_checkpoint(
            args.checkpoint,
            precision=max(0, args.precision),
            include_model=not args.exclude_model,
            as_json=args.json,
            head=args.head,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
