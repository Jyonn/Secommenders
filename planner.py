from __future__ import annotations

import argparse
import curses
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_CHOICES = [
    'mind',
    'recifvideo',
    'recifvideolarge',
    'recifvideoxlarge',
    'recifvideoxlargeall',
    'recifadsall',
    'recifadslargeall',
    'recifadsxlargeall',
    *[f'ra{scale}' for scale in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]],
    *[f'rv{scale}' for scale in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]],
    *[f'rvs{scale}' for scale in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]],
]

MODEL_CHOICES = [
    'scratch',
    'qwen35th08b',
    'qwen35th4b',
    'qwen35th9b',
    'llama3',
]

SOURCE_MODEL_CHOICES = [
    'pretrain-multimodal',
    'pretrain-text',
    'pretrain-vision',
    'llama3',
    'qwen3embedding06b',
]

TASK_CHOICES = ['uid', 'sid', 'hash', 'embedding']
HISTORY_CHOICES = [
    'uid',
    'sid',
    'hash',
    'embedding',
    'text',
    'uid+embedding',
    'uid+text',
    'sid+embedding',
    'sid+text',
    'hash+embedding',
]

SID_VARIANT_CHOICES = [
    'rqvae/coll',
    'rqvae/recon',
    'pqvae/coll',
    'opqvae/coll',
]

UID_VARIANT_CHOICES = [
    'flat',
    'hierarchical:auto:20',
    'hierarchical:auto:3,20',
    'hierarchical:20:3,20',
]

HASH_CODER_CHOICES = ['simhash', 'lsh', 'pcahash', 'itq']

GROUPS = {
    'schedule': {
        'effective_batch_size': 64,
        'poll_interval_seconds': 15,
        'priority': None,
        'batch_size_cap': None,
    },
    'trainer': {
        'main_metric': 'ndcg@10',
        'metrics': 'ndcg@5,ndcg@10,ndcg@20,hr@5,hr@10,hr@20,mrr',
        'epochs': 0,
        'learning_rate': 0.0001,
        'weight_decay': 0.01,
        'seed': 42,
        'batch_size': 64,
        'valid_only': False,
        'overwrite': 'auto',
        'patience': 3,
    },
    'representation': {
        'maxitems': 50,
        'item_text_max_tokens': 20,
        'model_max_length': None,
        'repr_combine': 'concat',
    },
    'sid': {
        'sid_embedding_model': None,
        'sid_codebook_size': 256,
        'sid_latent_dim': 64,
        'sid_hidden_dims': '2048,1024,512,256,128',
        'sid_num_quantizers': 3,
        'sid_num_codebooks': 3,
        'sid_assignment_strategy': 'sinkhorn',
        'sid_sinkhorn_epsilon': '0.0,0.0,0.003',
        'sid_epochs': 0,
        'sid_batch_size': 1000,
        'sid_lr': 0.001,
        'code_decoding': 'auto',
        'code_beam_width': 20,
        'code_beam_chunk_size': 0,
        'code_collision_loss_weight': 0.1,
    },
    'uid': {
        'uid_cluster_embedding_source': 'collaborative',
        'uid_cluster_content_model': None,
        'uid_cluster_content_reduce_dim': 128,
        'uid_cluster_normalize_blocks': True,
        'uid_cluster_mix_alpha': 0.5,
        'cluster_vector_size': 64,
        'cluster_window': 5,
        'cluster_patience': 5,
        'cluster_batch_size': 4096,
        'cluster_max_iter': 100,
        'cluster_n_init': 10,
    },
    'model': {
        'model_dtype': 'auto',
        'freeze_backbone': 'auto',
        'use_lora': 'auto',
        'lora_rank': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'lora_layers': None,
        'lora_target_modules': 'all-linear',
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
    },
    'backend': {
        'backend_uri': None,
        'backend_auth_token': None,
        'backend_uri_env': None,
        'backend_auth_env': None,
    },
    'notificator': {
        'notificator_name': None,
        'notificator_token': None,
        'notificator_bark': None,
    },
}


@dataclass
class PlanSelections:
    name: str
    output: Path
    datasets: list[str]
    models: list[str]
    targets: list[str]
    histories: list[str]
    source_models: list[str]
    sid_variants: list[str]
    uid_variants: list[str]
    hash_coders: list[str]
    params: dict[str, Any]


def _coerce_value(raw: str):
    text = raw.strip()
    lower = text.lower()
    if lower in {'none', 'null', ''}:
        return None
    if lower in {'true', 'yes', 'y'}:
        return True
    if lower in {'false', 'no', 'n'}:
        return False
    try:
        if any(ch in text for ch in ['.', 'e', 'E']):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _value_to_text(value: Any):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return ','.join(str(item) for item in value)
    return str(value)


def _read_models_from_dotfile(root: Path):
    path = root / '.model'
    if not path.exists():
        return []
    models = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        models.append(line.split('=', 1)[0].strip().lower())
    return models


def _draw_lines(stdscr, lines: list[str]):
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    for row, line in enumerate(lines[:height - 1]):
        stdscr.addnstr(row, 0, line, width - 1)
    stdscr.refresh()


def prompt_text(stdscr, title: str, default: str = ''):
    curses.echo()
    curses.curs_set(1)
    try:
        _draw_lines(stdscr, [title, '', f'default: {default}', 'input: '])
        value = stdscr.getstr(3, 7).decode('utf-8').strip()
    finally:
        curses.noecho()
        curses.curs_set(0)
    return value if value else default


def prompt_confirm(stdscr, title: str, default: bool = True):
    suffix = 'Y/n' if default else 'y/N'
    while True:
        value = prompt_text(stdscr, f'{title} ({suffix})', '')
        if not value:
            return default
        if value.lower() in {'y', 'yes'}:
            return True
        if value.lower() in {'n', 'no'}:
            return False


def _filtered_choices(choices: list[str], query: str):
    if not query:
        return list(choices)
    needle = query.lower()
    return [choice for choice in choices if needle in choice.lower()]


def select_choices(
    stdscr,
    title: str,
    choices: list[str],
    *,
    multi: bool,
    allow_custom: bool = True,
    selected: list[str] | None = None,
):
    selected_set = set(selected or [])
    cursor = 0
    offset = 0
    query = ''

    while True:
        visible = _filtered_choices(choices, query)
        if cursor >= len(visible):
            cursor = max(0, len(visible) - 1)
        height, width = stdscr.getmaxyx()
        page_size = max(5, height - 6)
        if cursor < offset:
            offset = cursor
        if cursor >= offset + page_size:
            offset = cursor - page_size + 1

        lines = [
            title,
            '↑/↓ move  space toggle  enter confirm  / filter  i manual  a all  c clear  q cancel',
            f'filter: {query or "-"}    selected: {len(selected_set)}',
            '',
        ]
        for index, choice in enumerate(visible[offset:offset + page_size], start=offset):
            marker = '●' if choice in selected_set else '○'
            pointer = '>' if index == cursor else ' '
            if not multi:
                marker = '●' if index == cursor else '○'
            lines.append(f'{pointer} {marker} {choice}')
        if not visible:
            lines.append('  no matches')
        _draw_lines(stdscr, lines)

        key = stdscr.getch()
        if key in {curses.KEY_UP, ord('k')}:
            cursor = max(0, cursor - 1)
        elif key in {curses.KEY_DOWN, ord('j')}:
            cursor = min(max(0, len(visible) - 1), cursor + 1)
        elif key == ord('/'):
            query = prompt_text(stdscr, f'Filter {title}', query)
            cursor = 0
            offset = 0
        elif key == ord('i') and allow_custom:
            raw = prompt_text(stdscr, f'Manual input for {title} (comma separated)', '')
            values = [part.strip().lower() for part in raw.split(',') if part.strip()]
            for value in values:
                if value not in choices:
                    choices.append(value)
                if multi:
                    selected_set.add(value)
                else:
                    return [value]
        elif key == ord('a') and multi:
            selected_set.update(visible)
        elif key == ord('c') and multi:
            selected_set.clear()
        elif key == ord(' ') and multi and visible:
            choice = visible[cursor]
            if choice in selected_set:
                selected_set.remove(choice)
            else:
                selected_set.add(choice)
        elif key in {10, 13}:
            if multi:
                return [choice for choice in choices if choice in selected_set]
            if visible:
                return [visible[cursor]]
        elif key == ord('q'):
            return [choice for choice in choices if choice in selected_set]


def edit_parameters(stdscr, params: dict[str, Any]):
    flat_items = []
    for group, values in GROUPS.items():
        for key in values:
            flat_items.append((group, key))
    cursor = 0
    offset = 0
    query = ''

    while True:
        visible = [
            (group, key)
            for group, key in flat_items
            if not query or query.lower() in group.lower() or query.lower() in key.lower()
        ]
        if cursor >= len(visible):
            cursor = max(0, len(visible) - 1)
        height, width = stdscr.getmaxyx()
        page_size = max(5, height - 7)
        if cursor < offset:
            offset = cursor
        if cursor >= offset + page_size:
            offset = cursor - page_size + 1

        lines = [
            'Edit Default Hyperparameters',
            '↑/↓ move  enter edit  / filter  d reset  q done',
            f'filter: {query or "-"}',
            '',
        ]
        for index, (group, key) in enumerate(visible[offset:offset + page_size], start=offset):
            pointer = '>' if index == cursor else ' '
            default_value = GROUPS[group][key]
            value = params.get(key, default_value)
            changed = '*' if value != default_value else ' '
            lines.append(f'{pointer} {changed} [{group}] {key} = {_value_to_text(value)}')
        if not visible:
            lines.append('  no matches')
        _draw_lines(stdscr, lines)

        key_code = stdscr.getch()
        if key_code in {curses.KEY_UP, ord('k')}:
            cursor = max(0, cursor - 1)
        elif key_code in {curses.KEY_DOWN, ord('j')}:
            cursor = min(max(0, len(visible) - 1), cursor + 1)
        elif key_code == ord('/'):
            query = prompt_text(stdscr, 'Filter hyperparameters', query)
            cursor = 0
            offset = 0
        elif key_code == ord('d') and visible:
            group, param_key = visible[cursor]
            params[param_key] = deepcopy(GROUPS[group][param_key])
        elif key_code in {10, 13} and visible:
            _, param_key = visible[cursor]
            old_value = params.get(param_key)
            raw = prompt_text(stdscr, f'Edit {param_key}', _value_to_text(old_value))
            params[param_key] = _coerce_value(raw)
        elif key_code == ord('q'):
            return params


def _parse_sid_variants(values: list[str]):
    variants = []
    for value in values:
        if '/' in value:
            coder, export = value.split('/', 1)
        elif ':' in value:
            coder, export = value.split(':', 1)
        else:
            coder, export = value, 'coll'
        variants.append((coder.strip().lower(), export.strip().lower()))
    return variants


def _parse_uid_variants(values: list[str]):
    variants = []
    for value in values:
        parts = [part.strip() for part in value.split(':')]
        decoding = parts[0].lower()
        if decoding == 'flat':
            variants.append('flat')
        elif decoding == 'hierarchical':
            levels = parts[1] if len(parts) > 1 and parts[1] else 'auto'
            topk = parts[2] if len(parts) > 2 and parts[2] else '20'
            variants.append(('hierarchical', levels, topk))
        else:
            variants.append(decoding)
    return variants


def _history_uses_semantic(histories: list[str], targets: list[str]):
    used = set(targets)
    for history in histories:
        used.update(part.strip().lower() for part in history.split('+') if part.strip())
    return bool(used & {'sid', 'hash', 'embedding'})


def _compact_args(params: dict[str, Any]):
    excluded = {
        'effective_batch_size',
        'poll_interval_seconds',
        'priority',
        'batch_size_cap',
        'backend_uri',
        'backend_auth_token',
        'backend_uri_env',
        'backend_auth_env',
        'notificator_name',
        'notificator_token',
        'notificator_bark',
    }
    defaults = {}
    for values in GROUPS.values():
        defaults.update(values)
    args = {}
    for key, value in params.items():
        if key in excluded or value is None:
            continue
        if value == defaults.get(key):
            continue
        args[key] = value
    return args


def build_payload(selection: PlanSelections):
    from utils import Schedule

    params = selection.params
    schedule = Schedule(
        name=selection.name,
        poll_interval_seconds=int(params['poll_interval_seconds']),
        effective_batch_size=int(params['effective_batch_size']),
        backend_host=params.get('backend_uri'),
        backend_auth=params.get('backend_auth_token'),
        backend_host_env=params.get('backend_uri_env'),
        backend_auth_env=params.get('backend_auth_env'),
    )
    schedule_args = _compact_args(params)
    schedule.defaults(**schedule_args)
    if params.get('priority') is not None or params.get('batch_size_cap') is not None:
        schedule.plan_defaults(
            priority=params.get('priority'),
            batch_size_cap=params.get('batch_size_cap'),
        )

    source_models = selection.source_models if _history_uses_semantic(selection.histories, selection.targets) else None
    schedule.grid(
        selection.name,
        datasets=selection.datasets,
        models=selection.models,
        targets=selection.targets,
        histories=selection.histories,
        source_models=source_models,
        sid_variants=_parse_sid_variants(selection.sid_variants),
        uid_variants=_parse_uid_variants(selection.uid_variants),
        hash_coders=selection.hash_coders,
    )
    payload = schedule.to_dict(path=selection.output)

    notificator = {
        'name': params.get('notificator_name'),
        'token': params.get('notificator_token'),
        'bark': params.get('notificator_bark'),
    }
    if all(notificator.values()):
        payload['notificator'] = notificator
    return payload


def write_plan(selection: PlanSelections):
    payload = build_payload(selection)
    selection.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    except ModuleNotFoundError:
        text = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    selection.output.write_text(text)
    return payload


def preview_payload(stdscr, selection: PlanSelections):
    try:
        payload = build_payload(selection)
        lines = [
            'Plan Preview',
            '',
            f'name: {payload["name"]}',
            f'output: {selection.output}',
            f'experiments: {len(payload["experiments"])}',
            f'backend: {"yes" if payload.get("backend") else "no"}',
            f'notificator: {"yes" if payload.get("notificator") else "no"}',
            '',
            'first experiments:',
        ]
        for exp in payload['experiments'][:8]:
            lines.append(f'  - {exp["name"]}')
        if len(payload['experiments']) > 8:
            lines.append(f'  ... {len(payload["experiments"]) - 8} more')
        lines.extend(['', 'press any key'])
    except Exception as exc:
        lines = ['Plan Preview Failed', '', repr(exc), '', 'press any key']
    _draw_lines(stdscr, lines)
    stdscr.getch()


def collect_selection(stdscr, args):
    curses.curs_set(0)
    root = Path(__file__).resolve().parent
    model_choices = sorted(set(MODEL_CHOICES + _read_models_from_dotfile(root)))

    name = prompt_text(stdscr, 'Schedule name', args.name or 'interactive')
    output = Path(prompt_text(stdscr, 'Output yaml path', args.output or f'config/{name}_scheduler.yaml'))

    datasets = select_choices(
        stdscr,
        'Select datasets',
        list(DATASET_CHOICES),
        multi=True,
        selected=args.datasets.split(',') if args.datasets else [],
    )
    models = select_choices(stdscr, 'Select models', model_choices, multi=True, selected=args.models.split(',') if args.models else [])
    targets = select_choices(stdscr, 'Select task_type targets', list(TASK_CHOICES), multi=True, selected=args.targets.split(',') if args.targets else ['uid'])
    histories = select_choices(
        stdscr,
        'Select history representations',
        list(HISTORY_CHOICES),
        multi=True,
        selected=args.histories.split(',') if args.histories else ['uid'],
    )

    source_models = []
    if _history_uses_semantic(histories, targets):
        source_models = select_choices(
            stdscr,
            'Select semantic/source embedding models',
            list(SOURCE_MODEL_CHOICES),
            multi=True,
            selected=args.source_models.split(',') if args.source_models else ['pretrain-multimodal'],
        )

    sid_variants = ['rqvae/coll']
    if 'sid' in set(targets) or any('sid' in history.split('+') for history in histories):
        sid_variants = select_choices(
            stdscr,
            'Select SID variants as coder/export',
            list(SID_VARIANT_CHOICES),
            multi=True,
            selected=args.sid_variants.split(',') if args.sid_variants else ['rqvae/coll'],
        )

    uid_variants = ['flat']
    if 'uid' in targets:
        uid_variants = select_choices(
            stdscr,
            'Select UID decoding variants',
            list(UID_VARIANT_CHOICES),
            multi=True,
            selected=args.uid_variants.split(',') if args.uid_variants else ['flat'],
        )

    hash_coders = ['simhash']
    if 'hash' in set(targets) or any('hash' in history.split('+') for history in histories):
        hash_coders = select_choices(
            stdscr,
            'Select hash coders',
            list(HASH_CODER_CHOICES),
            multi=True,
            selected=args.hash_coders.split(',') if args.hash_coders else ['simhash'],
        )

    params = {}
    for values in GROUPS.values():
        params.update(deepcopy(values))
    edit_parameters(stdscr, params)

    selection = PlanSelections(
        name=name,
        output=output,
        datasets=datasets,
        models=models,
        targets=targets,
        histories=histories,
        source_models=source_models,
        sid_variants=sid_variants,
        uid_variants=uid_variants,
        hash_coders=hash_coders,
        params=params,
    )
    preview_payload(stdscr, selection)
    if not prompt_confirm(stdscr, f'Write plan to {selection.output}?', True):
        raise SystemExit('cancelled')
    return selection


def build_parser():
    parser = argparse.ArgumentParser(description='Interactive scheduler plan builder.')
    parser.add_argument('--name', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--datasets', default=None, help='Optional comma-separated preselection.')
    parser.add_argument('--models', default=None, help='Optional comma-separated preselection.')
    parser.add_argument('--targets', default=None, help='Optional comma-separated preselection.')
    parser.add_argument('--histories', default=None, help='Optional comma-separated preselection.')
    parser.add_argument('--source-models', default=None, help='Optional comma-separated preselection.')
    parser.add_argument('--sid-variants', default=None, help='Optional comma-separated coder/export preselection.')
    parser.add_argument('--uid-variants', default=None, help='Optional comma-separated decoding preselection.')
    parser.add_argument('--hash-coders', default=None, help='Optional comma-separated hash coder preselection.')
    parser.add_argument('--dump-defaults', action='store_true', help='Print editable parameter defaults as JSON and exit.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.dump_defaults:
        print(json.dumps(GROUPS, indent=2, ensure_ascii=False))
        return
    selection = curses.wrapper(lambda stdscr: collect_selection(stdscr, args))
    payload = write_plan(selection)
    print(f'wrote {len(payload["experiments"])} experiments to {selection.output}')


if __name__ == '__main__':
    main()
