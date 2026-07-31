from __future__ import annotations

import argparse
import curses
import json
import string
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_CHOICES = ['scratch', 'qwen35th08b', 'qwen35th4b', 'qwen35th9b', 'llama3']
SOURCE_MODEL_CHOICES = ['pretrain-multimodal', 'pretrain-text', 'pretrain-vision', 'llama3', 'qwen3embedding06b']

REPRESENTATION_PAIR_CHOICES = [
    'uid2uid',
    'sid2sid',
    'hash2hash',
    'embedding2embedding',
    'uid+embedding2uid',
    'uid+text2uid',
    'uid+sid2uid',
    'uid+hash2uid',
    'sid+embedding2sid',
    'sid+text2sid',
    'sid+uid2sid',
    'hash+embedding2hash',
    'embedding+uid2embedding',
]

SID_VARIANT_CHOICES = ['rqvae/coll', 'rqvae/recon', 'pqvae/coll', 'opqvae/coll']
UID_VARIANT_CHOICES = ['flat', 'hierarchical:auto:20', 'hierarchical:auto:3,20', 'hierarchical:20:3,20']
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
        'lr_scheduler': 'constant',
        'warmup_ratio': 0.0,
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
    'hash': {
        'hash_embedding_model': None,
        'hash_num_bits': 24,
        'hash_num_tables': 1,
        'hash_projection_distribution': 'gaussian',
        'hash_use_median_thresholds': True,
        'hash_num_iterations': 50,
        'hash_normalize_inputs': True,
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
class ChoiceStep:
    key: str
    title: str
    choices: list[str]
    selected: list[str]
    multi: bool = True
    required: bool = True
    allow_custom: bool = True
    query: str = ''
    cursor: int = 0
    offset: int = 0
    hint: str = ''


@dataclass
class PlanSelections:
    name: str
    output: Path
    datasets: list[str]
    models: list[str]
    representation_pairs: list[str]
    source_models: list[str]
    sid_variants: list[str]
    uid_variants: list[str]
    hash_coders: list[str]
    seeds: list[int]
    params: dict[str, Any]


def _defaults():
    params = {}
    for values in GROUPS.values():
        params.update(deepcopy(values))
    return params


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


def _dataset_sort_key(name: str):
    recif_base_order = {
        'recifvideo': 10,
        'recifvideolarge': 11,
        'recifvideoxlarge': 12,
        'recifvideoall': 13,
        'recifvideolargeall': 14,
        'recifvideoxlargeall': 15,
        'recifvideoxlargeallofficial': 16,
        'rvf': 17,
        'recifadsall': 20,
        'recifadslargeall': 21,
        'recifadsxlargeall': 22,
        'raf': 23,
    }
    base_order = {
        'mind': 0,
        'mindf': 0.5,
        'movielens': 1,
        'goodreads': 2,
        'yelp': 3,
        'cds': 4,
        'hm': 5,
        'pens': 6,
        'microlens': 7,
    }
    if name in base_order:
        return (0, base_order[name], 0, name)
    if name in recif_base_order:
        return (1, recif_base_order[name], 0, name)
    if name.startswith('minds') and name[5:].isdigit():
        return (0, base_order['mind'], int(name[5:]), name)
    for prefix_index, prefix in enumerate(('ra', 'rv', 'ras', 'rvs', 'rvt'), start=2):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return (prefix_index, int(name[len(prefix):]), 0, name)
    return (9, 0, 0, name)


def _available_dataset_choices():
    from utils.class_hub import ClassHub

    return sorted(ClassHub.formatters().class_dict, key=_dataset_sort_key)


def _parse_csv(value: str | None):
    if not value:
        return []
    return [part.strip().lower() for part in value.split(',') if part.strip()]


def _parse_int_csv(value: str | None, default: list[int] | None = None):
    values = _parse_csv(value)
    if not values:
        return list(default or [])
    return [int(value) for value in values]


def _ensure_color():
    if not curses.has_colors():
        return
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_RED, -1)
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    curses.init_pair(7, curses.COLOR_BLUE, -1)
    curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_GREEN)
    curses.init_pair(10, curses.COLOR_BLACK, curses.COLOR_YELLOW)


def _attr(pair: int, fallback=0):
    if curses.has_colors():
        return curses.color_pair(pair)
    return fallback


def _safe_add(stdscr, row: int, col: int, text: str, attr=0):
    height, width = stdscr.getmaxyx()
    if row < 0 or row >= height or col >= width:
        return
    stdscr.addnstr(row, col, text, max(0, width - col - 1), attr)


def _safe_add_segments(stdscr, row: int, col: int, segments: list[tuple[str, int]]):
    height, width = stdscr.getmaxyx()
    if row < 0 or row >= height:
        return
    current = col
    for text, attr in segments:
        if current >= width:
            return
        stdscr.addnstr(row, current, text, max(0, width - current - 1), attr)
        current += len(text)


def _pill_attr(pair: int):
    return _attr(pair, curses.A_REVERSE) | curses.A_BOLD


def _draw_footer_hints(stdscr, hints: list[tuple[str, str, int]], message: str = ''):
    height, width = stdscr.getmaxyx()
    if message:
        _safe_add(stdscr, height - 4, 2, message, _attr(4, curses.A_BOLD))
    _safe_add(stdscr, height - 2, 0, ' ' * max(0, width - 1), _attr(7))
    col = 2
    for key, label, pair in hints:
        segments = [
            (f' {key} ', _pill_attr(pair)),
            (f' {label}  ', _attr(3)),
        ]
        _safe_add_segments(stdscr, height - 2, col, segments)
        col += len(key) + len(label) + 6
        if col >= width - 8:
            break


def _key_f(index: int):
    named_key = getattr(curses, f'KEY_F{index}', None)
    if named_key is not None:
        return named_key
    return getattr(curses, 'KEY_F0', 264) + int(index)


def prompt_text(stdscr, title: str, default: str = ''):
    curses.echo()
    curses.curs_set(1)
    try:
        stdscr.erase()
        _safe_add(stdscr, 1, 2, title, _attr(1, curses.A_BOLD))
        _safe_add(stdscr, 3, 2, f'default: {default or "-"}', _attr(3))
        _safe_add(stdscr, 5, 2, 'input: ')
        stdscr.refresh()
        value = stdscr.getstr(5, 9).decode('utf-8').strip()
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


def _visible_choices(step: ChoiceStep):
    if not step.query:
        return list(step.choices)
    needle = step.query.lower()
    return [choice for choice in step.choices if needle in choice.lower()]


def _selected_ordered(step: ChoiceStep):
    selected = set(step.selected)
    return [choice for choice in step.choices if choice in selected]


def _step_valid(step: ChoiceStep):
    return not step.required or bool(step.selected)


def _wizard_validation_error(step: ChoiceStep, steps: list[ChoiceStep]):
    if not _step_valid(step):
        return f'{step.title} is required before continuing.'
    if step.key == 'representation_pairs':
        for pair in step.selected:
            try:
                _parse_representation_pair(pair)
            except ValueError as exc:
                return str(exc)
    return None


def _draw_chrome(stdscr, steps: list[ChoiceStep], index: int, message: str = ''):
    height, width = stdscr.getmaxyx()
    stdscr.erase()
    title = ' Secommenders Schedule Planner '
    _safe_add(stdscr, 0, 0, title.ljust(width - 1), _attr(8, curses.A_REVERSE) | curses.A_BOLD)
    _safe_add(stdscr, 0, max(0, width - 16), f' step {index + 1}/{len(steps)} ', _pill_attr(5))
    progress_segments = []
    for i, step in enumerate(steps):
        mark = '✓' if _step_valid(step) else '!'
        label = f' {mark} {step.title} '
        if i == index:
            progress_segments.append((label, _pill_attr(5)))
        elif _step_valid(step):
            progress_segments.append((label, _attr(2, curses.A_BOLD)))
        else:
            progress_segments.append((label, _attr(4, curses.A_BOLD)))
        progress_segments.append(('  ', 0))
    _safe_add_segments(stdscr, 2, 2, progress_segments)
    _draw_footer_hints(
        stdscr,
        [
            ('type', 'filter', 8),
            ('↑↓', 'move', 5),
            ('space', 'toggle', 9),
            ('enter/→', 'next', 2),
            ('←', 'prev', 6),
            ('F2', 'manual', 10),
            ('F3', 'all', 9),
            ('F4', 'clear', 4),
            ('bksp', 'edit filter', 3),
        ],
        message=message,
    )


def _draw_choice_step(stdscr, step: ChoiceStep, steps: list[ChoiceStep], index: int, message: str = ''):
    _draw_chrome(stdscr, steps, index, message)
    height, width = stdscr.getmaxyx()
    visible = _visible_choices(step)
    window_size = min(10, max(4, height - 11))
    if step.cursor >= len(visible):
        step.cursor = max(0, len(visible) - 1)
    if step.cursor < step.offset:
        step.offset = step.cursor
    if step.cursor >= step.offset + window_size:
        step.offset = step.cursor - window_size + 1

    _safe_add(stdscr, 4, 2, step.title, _attr(1, curses.A_BOLD))
    _safe_add(stdscr, 5, 2, step.hint, _attr(7))
    _safe_add_segments(
        stdscr,
        6,
        2,
        [
            (' filter ', _pill_attr(8)),
            (f' {step.query or "type to narrow"} ', _attr(3)),
        ],
    )
    _safe_add_segments(
        stdscr,
        6,
        max(30, width - 30),
        [
            (' selected ', _pill_attr(9)),
            (f' {len(step.selected)} ', _attr(2, curses.A_BOLD)),
        ],
    )

    top = 8
    if not visible:
        _safe_add(stdscr, top, 4, 'No matches. Type a different filter or press F2 for manual input.', _attr(4))
    for row, choice_index in enumerate(range(step.offset, min(len(visible), step.offset + window_size)), start=top):
        choice = visible[choice_index]
        active = choice_index == step.cursor
        selected = choice in set(step.selected)
        pointer = '❯' if active else ' '
        marker = '●' if selected else '○'
        attr = _attr(5) if active else (_attr(2) if selected else 0)
        _safe_add(stdscr, row, 4, f'{pointer} {marker} {choice:<36}', attr)

    footer = f'{step.offset + 1 if visible else 0}-{min(len(visible), step.offset + window_size)} / {len(visible)}'
    _safe_add_segments(stdscr, top + window_size + 1, 4, [(' showing ', _pill_attr(8)), (f' {footer}', _attr(3))])
    if step.selected:
        preview = ', '.join(step.selected[:6])
        if len(step.selected) > 6:
            preview += f', ... +{len(step.selected) - 6}'
        _safe_add(stdscr, top + window_size + 2, 4, f'selected: {preview}', _attr(2))
    stdscr.refresh()


def _manual_add(stdscr, step: ChoiceStep):
    raw = prompt_text(stdscr, f'Manual input for {step.title} (comma separated)', '')
    values = [part.strip().lower() for part in raw.split(',') if part.strip()]
    for value in values:
        if value not in step.choices:
            step.choices.append(value)
        if step.multi:
            if value not in step.selected:
                step.selected.append(value)
        else:
            step.selected = [value]


def _handle_choice_key(stdscr, step: ChoiceStep, key: int):
    visible = _visible_choices(step)
    if key == curses.KEY_UP:
        step.cursor = max(0, step.cursor - 1)
    elif key == curses.KEY_DOWN:
        step.cursor = min(max(0, len(visible) - 1), step.cursor + 1)
    elif key == curses.KEY_NPAGE:
        step.cursor = min(max(0, len(visible) - 1), step.cursor + 10)
    elif key == curses.KEY_PPAGE:
        step.cursor = max(0, step.cursor - 10)
    elif key in {curses.KEY_BACKSPACE, 127, 8}:
        step.query = step.query[:-1]
        step.cursor = 0
        step.offset = 0
    elif key == _key_f(2) and step.allow_custom:
        _manual_add(stdscr, step)
    elif key in {_key_f(3), 1} and step.multi:
        for choice in visible:
            if choice not in step.selected:
                step.selected.append(choice)
    elif key in {_key_f(4), 24} and step.multi:
        step.selected = []
    elif key == ord(' ') and visible:
        choice = visible[step.cursor]
        if step.multi:
            if choice in step.selected:
                step.selected.remove(choice)
            else:
                step.selected.append(choice)
        else:
            step.selected = [choice]
    elif 0 <= key <= 255 and chr(key) in string.printable and chr(key) not in {'\n', '\r', '\t'}:
        char = chr(key)
        if char == '/':
            step.query = ''
        elif char != ' ':
            step.query += char
        step.cursor = 0
        step.offset = 0


def run_choice_wizard(stdscr, steps: list[ChoiceStep]):
    index = 0
    message = ''
    while 0 <= index < len(steps):
        step = steps[index]
        _draw_choice_step(stdscr, step, steps, index, message)
        message = ''
        key = stdscr.getch()
        if key == curses.KEY_LEFT:
            index = max(0, index - 1)
        elif key in {curses.KEY_RIGHT, 10, 13, 9}:
            error = _wizard_validation_error(step, steps)
            if error:
                message = error
                continue
            index += 1
        elif key == 27:
            raise SystemExit('cancelled')
        else:
            _handle_choice_key(stdscr, step, key)
    return {step.key: _selected_ordered(step) for step in steps}


def edit_parameters(stdscr, params: dict[str, Any], groups: list[str] | None = None):
    groups = list(groups or GROUPS)
    group_index = 0
    cursors = {group: 0 for group in groups}
    offsets = {group: 0 for group in groups}
    query = ''
    message = ''

    while True:
        group = groups[group_index]
        keys = list(GROUPS[group])
        visible = [key for key in keys if not query or query.lower() in key.lower() or query.lower() in group.lower()]
        cursor = cursors[group]
        if cursor >= len(visible):
            cursor = max(0, len(visible) - 1)
        height, width = stdscr.getmaxyx()
        page_size = min(10, max(4, height - 11))
        offset = offsets[group]
        if cursor < offset:
            offset = cursor
        if cursor >= offset + page_size:
            offset = cursor - page_size + 1
        cursors[group] = cursor
        offsets[group] = offset

        stdscr.erase()
        _safe_add(stdscr, 0, 0, ' Hyperparameter Groups '.ljust(width - 1), _attr(8, curses.A_REVERSE) | curses.A_BOLD)
        _safe_add(stdscr, 0, max(0, width - 18), f' group {group_index + 1}/{len(groups)} ', _pill_attr(5))
        tab_segments = []
        for i, name in enumerate(groups):
            changed = any(params.get(key) != GROUPS[name][key] for key in GROUPS[name])
            label = f' {name}{"*" if changed else ""} '
            if i == group_index:
                tab_segments.append((label, _pill_attr(5)))
            elif changed:
                tab_segments.append((label, _attr(2, curses.A_BOLD)))
            else:
                tab_segments.append((label, _attr(7)))
            tab_segments.append((' ', 0))
        _safe_add_segments(stdscr, 2, 2, tab_segments)
        _safe_add_segments(
            stdscr,
            4,
            2,
            [
                (' filter ', _pill_attr(8)),
                (f' {query or "type to narrow"} ', _attr(3)),
                (' group ', _pill_attr(6)),
                (f' {group} ', _attr(6, curses.A_BOLD)),
            ],
        )
        _draw_footer_hints(
            stdscr,
            [
                ('type', 'filter', 8),
                ('←→', 'group', 6),
                ('↑↓', 'move', 5),
                ('enter', 'edit value', 9),
                ('F5', 'reset', 10),
                ('F10', 'done', 2),
                ('bksp', 'edit filter', 3),
            ],
            message=message,
        )

        top = 6
        _safe_add_segments(
            stdscr,
            top - 1,
            4,
            [
                (' key ', _pill_attr(8)),
                (' ' * 31, 0),
                (' value ', _pill_attr(9)),
                ('   default ', _pill_attr(7)),
            ],
        )
        for row, key_index in enumerate(range(offset, min(len(visible), offset + page_size)), start=top):
            param_key = visible[key_index]
            active = key_index == cursor
            default_value = GROUPS[group][param_key]
            value = params.get(param_key, default_value)
            changed = '*' if value != default_value else ' '
            attr = _attr(5) if active else (_attr(2) if changed == '*' else 0)
            pointer = '❯' if active else ' '
            _safe_add(stdscr, row, 4, f'{pointer} {changed} {param_key:<32}', attr)
            _safe_add(stdscr, row, 42, f'{_value_to_text(value):<24}', _attr(2) if changed == '*' and not active else attr)
            _safe_add(stdscr, row, 68, _value_to_text(default_value), _attr(7))
        if not visible:
            _safe_add(stdscr, top, 4, 'No matching parameters.', _attr(4))
        stdscr.refresh()

        key_code = stdscr.getch()
        message = ''
        if key_code == curses.KEY_LEFT:
            group_index = max(0, group_index - 1)
        elif key_code == curses.KEY_RIGHT:
            group_index = min(len(groups) - 1, group_index + 1)
        elif key_code == curses.KEY_UP:
            cursors[group] = max(0, cursor - 1)
        elif key_code == curses.KEY_DOWN:
            cursors[group] = min(max(0, len(visible) - 1), cursor + 1)
        elif key_code in {curses.KEY_BACKSPACE, 127, 8}:
            query = query[:-1]
            cursors[group] = 0
            offsets[group] = 0
        elif key_code == _key_f(5) and visible:
            params[visible[cursor]] = deepcopy(GROUPS[group][visible[cursor]])
        elif key_code in {10, 13} and visible:
            param_key = visible[cursor]
            raw = prompt_text(stdscr, f'Edit {param_key}', _value_to_text(params.get(param_key)))
            params[param_key] = _coerce_value(raw)
        elif key_code in {_key_f(10), 7}:
            return params
        elif 0 <= key_code <= 255 and chr(key_code) in string.printable and chr(key_code) not in {'\n', '\r', '\t'}:
            char = chr(key_code)
            if char == '/':
                query = ''
            elif char != ' ':
                query += char
            cursors[group] = 0
            offsets[group] = 0


def _parse_representation_pair(value: str):
    text = value.strip().lower()
    if '2' not in text:
        raise ValueError(f'Invalid representation pair: {value}')
    history_text, target = text.rsplit('2', 1)
    history = tuple(part.strip() for part in history_text.split('+') if part.strip())
    if not history or not target:
        raise ValueError(f'Invalid representation pair: {value}')
    return target, history


def _used_views(representation_pairs: list[str]):
    used = set()
    for pair in representation_pairs:
        target, history = _parse_representation_pair(pair)
        used.add(target)
        used.update(history)
    return used


def _target_views(representation_pairs: list[str]):
    return {_parse_representation_pair(pair)[0] for pair in representation_pairs}


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
        'seed',
    }
    defaults = _defaults()
    args = {}
    for key, value in params.items():
        if key in excluded or value is None:
            continue
        if value == defaults.get(key):
            continue
        args[key] = value
    return args


def _compact_common_args(params: dict[str, Any]):
    args = _compact_args(params)
    view_specific_keys = (
        _keys_for_group('sid')
        | _keys_for_group('hash')
        | _keys_for_group('uid')
        | {
            'uid_decoding',
            'uid_cluster_levels',
            'uid_cluster_topk',
            'repr_source_model',
            'sid_coder',
            'sid_export',
            'hash_coder',
        }
    )
    for key in view_specific_keys:
        args.pop(key, None)
    return args


def _keys_for_group(group: str):
    return set(GROUPS.get(group, {}))


def _compact_args_for_experiment(params: dict[str, Any], *, target: str, history: tuple[str, ...]):
    args = _compact_args(params)
    used_views = set(history) | {target}

    if 'sid' not in used_views:
        for key in _keys_for_group('sid'):
            args.pop(key, None)
    elif target != 'sid':
        for key in ('code_decoding', 'code_beam_width', 'code_beam_chunk_size'):
            args.pop(key, None)

    if 'hash' not in used_views:
        for key in _keys_for_group('hash'):
            args.pop(key, None)

    if target != 'uid':
        for key in _keys_for_group('uid'):
            args.pop(key, None)
        args.pop('uid_decoding', None)
        args.pop('uid_cluster_levels', None)
        args.pop('uid_cluster_topk', None)
    elif str(args.get('uid_decoding', 'flat')).strip().lower() != 'hierarchical':
        for key in _keys_for_group('uid'):
            args.pop(key, None)
        args.pop('uid_cluster_levels', None)
        args.pop('uid_cluster_topk', None)

    if not any(view in {'sid', 'hash'} for view in used_views):
        args.pop('code_collision_loss_weight', None)
    if 'text' not in used_views:
        args.pop('item_text_max_tokens', None)
    if len(used_views) <= 1:
        args.pop('repr_combine', None)
    return args


def _group_pairs_by_target(representation_pairs: list[str]):
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for pair in representation_pairs:
        target, history = _parse_representation_pair(pair)
        grouped.setdefault(target, []).append(history)
    return grouped


def _is_scratch_model(model: str):
    return str(model).strip().lower() in {'scratch', 'scratchlegacy'}


def _models_compatible_with_history(models: list[str], history: tuple[str, ...]):
    if 'text' not in history:
        return list(models)
    return [model for model in models if not _is_scratch_model(model)]


def _seed_suffix(seed: int, seeds: list[int]):
    return f'_s{seed}' if len(seeds) > 1 else ''


def _history_label(history: tuple[str, ...]):
    return '+'.join(history)


def _editable_groups_for_selection(
    representation_pairs: list[str],
    uid_variants: list[str],
):
    used_views = _used_views(representation_pairs)
    target_views = _target_views(representation_pairs)
    groups = ['schedule', 'trainer', 'representation']
    if 'sid' in used_views:
        groups.append('sid')
    if 'hash' in used_views:
        groups.append('hash')
    parsed_uid_variants = _parse_uid_variants(uid_variants or [])
    uses_hierarchical_uid = any(
        isinstance(variant, tuple) and variant and variant[0] == 'hierarchical'
        for variant in parsed_uid_variants
    )
    if 'uid' in target_views and uses_hierarchical_uid:
        groups.append('uid')
    groups.extend(['model', 'backend', 'notificator'])
    return groups


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
    schedule.defaults(**_compact_common_args(params))
    if params.get('priority') is not None or params.get('batch_size_cap') is not None:
        schedule.plan_defaults(priority=params.get('priority'), batch_size_cap=params.get('batch_size_cap'))

    skipped = []
    seeds = selection.seeds or [42]
    generated_studies = 0
    for pair in selection.representation_pairs:
        target, history = _parse_representation_pair(pair)
        used_views = set(history) | {target}
        models = _models_compatible_with_history(selection.models, history)
        excluded_models = [model for model in selection.models if model not in models]
        if excluded_models:
            skipped.append(f'{pair}: skipped text-incompatible models {", ".join(excluded_models)}')
        if not models:
            skipped.append(f'{pair}: no compatible model after excluding scratch text pairs')
            continue
        source_models = selection.source_models if used_views & {'sid', 'hash', 'embedding'} else None
        sid_variants = _parse_sid_variants(selection.sid_variants) if 'sid' in used_views else None
        hash_coders = selection.hash_coders if 'hash' in used_views else None
        uid_variants = _parse_uid_variants(selection.uid_variants) if target == 'uid' else None
        base_args = _compact_args_for_experiment(params, target=target, history=history)
        for seed in seeds:
            args = deepcopy(base_args)
            if int(seed) != 42 or len(seeds) > 1:
                args['seed'] = int(seed)
            schedule.grid(
                f'{selection.name}{_seed_suffix(int(seed), seeds)}',
                datasets=selection.datasets,
                models=models,
                targets=[target],
                histories=[history],
                args=args,
                source_models=source_models,
                sid_variants=sid_variants,
                uid_variants=uid_variants,
                hash_coders=hash_coders,
            )
            generated_studies += 1
    if not generated_studies:
        detail = '; '.join(skipped) if skipped else 'no representation pairs selected'
        raise ValueError(f'No valid planner experiments generated: {detail}')
    payload = schedule.to_dict(path=selection.output)
    if skipped:
        payload['planner_warnings'] = skipped

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
            f'seeds: {", ".join(str(seed) for seed in selection.seeds)}',
            f'pairs: {", ".join(selection.representation_pairs)}',
            f'backend: {"yes" if payload.get("backend") else "no"}',
            f'notificator: {"yes" if payload.get("notificator") else "no"}',
            '',
            'first experiments:',
        ]
        for exp in payload['experiments'][:8]:
            lines.append(f'  - {exp["name"]}')
        if len(payload['experiments']) > 8:
            lines.append(f'  ... {len(payload["experiments"]) - 8} more')
        if payload.get('planner_warnings'):
            lines.extend(['', 'warnings:'])
            for warning in payload['planner_warnings'][:4]:
                lines.append(f'  - {warning}')
        lines.extend(['', 'press any key'])
        attr = _attr(2)
    except Exception as exc:
        lines = ['Plan Preview Failed', '', repr(exc), '', 'press any key']
        attr = _attr(4)

    stdscr.erase()
    for row, line in enumerate(lines):
        _safe_add(stdscr, row + 1, 2, line, attr if row == 0 else 0)
    stdscr.refresh()
    stdscr.getch()


def _dynamic_steps(args, model_choices: list[str]):
    base_steps = [
        ChoiceStep(
            key='datasets',
            title='Datasets',
            choices=_available_dataset_choices(),
            selected=_parse_csv(args.datasets),
            hint='Choose one or more datasets. Type "ra", "rvs", or "rvt" to narrow RecIF scales.',
        ),
        ChoiceStep(
            key='models',
            title='Models',
            choices=model_choices,
            selected=_parse_csv(args.models),
            hint='Choose scratch or LLM backbones. Manual aliases from .model are supported.',
        ),
        ChoiceStep(
            key='representation_pairs',
            title='Representation Pairs',
            choices=list(REPRESENTATION_PAIR_CHOICES),
            selected=_parse_csv(args.representations) or _pairs_from_legacy_args(args),
            hint='Each option is atomic, e.g. uid2uid or sid2sid. No task/history cross product is created.',
        ),
        ChoiceStep(
            key='seeds',
            title='Seeds',
            choices=['42', '0', '1', '2', '3', '4', '5', '2024', '2025'],
            selected=[str(seed) for seed in _parse_int_csv(args.seeds, default=[42])],
            hint='Choose one or more random seeds. Press F2 to enter custom comma-separated seeds.',
        ),
    ]
    return base_steps


def _pairs_from_legacy_args(args):
    targets = _parse_csv(args.targets)
    histories = _parse_csv(args.histories)
    if not targets or not histories:
        return []
    pairs = []
    for target in targets:
        for history in histories:
            pairs.append(f'{history}2{target}')
    return pairs


def _conditional_steps(args, selected: dict[str, list[str]]):
    views = _used_views(selected['representation_pairs'])
    steps = []
    if views & {'sid', 'hash', 'embedding'}:
        steps.append(
            ChoiceStep(
                key='source_models',
                title='Source Models',
                choices=list(SOURCE_MODEL_CHOICES),
                selected=_parse_csv(args.source_models) or ['pretrain-multimodal'],
                hint='Required for sid/hash/embedding views.',
            )
        )
    if 'sid' in views:
        steps.append(
            ChoiceStep(
                key='sid_variants',
                title='SID Variants',
                choices=list(SID_VARIANT_CHOICES),
                selected=_parse_csv(args.sid_variants) or ['rqvae/coll'],
                hint='Format: coder/export. Press F2 to type custom values.',
            )
        )
    if any(_parse_representation_pair(pair)[0] == 'uid' for pair in selected['representation_pairs']):
        steps.append(
            ChoiceStep(
                key='uid_variants',
                title='UID Decoding',
                choices=list(UID_VARIANT_CHOICES),
                selected=_parse_csv(args.uid_variants) or ['flat'],
                hint='Format: flat or hierarchical:levels:topk.',
            )
        )
    if 'hash' in views:
        steps.append(
            ChoiceStep(
                key='hash_coders',
                title='Hash Coders',
                choices=list(HASH_CODER_CHOICES),
                selected=_parse_csv(args.hash_coders) or ['simhash'],
                hint='Required when any pair uses hash.',
            )
        )
    return steps


def collect_selection(stdscr, args):
    curses.curs_set(0)
    _ensure_color()
    root = Path(__file__).resolve().parent
    model_choices = sorted(set(MODEL_CHOICES + _read_models_from_dotfile(root)))

    name = prompt_text(stdscr, 'Schedule name', args.name or 'interactive')
    output = Path(prompt_text(stdscr, 'Output yaml path', args.output or f'config/{name}_scheduler.yaml'))

    primary = run_choice_wizard(stdscr, _dynamic_steps(args, model_choices))
    conditional = run_choice_wizard(stdscr, _conditional_steps(args, primary))

    params = _defaults()
    editable_groups = _editable_groups_for_selection(
        primary['representation_pairs'],
        conditional.get('uid_variants', ['flat']),
    )
    edit_parameters(stdscr, params, editable_groups)

    selection = PlanSelections(
        name=name,
        output=output,
        datasets=primary['datasets'],
        models=primary['models'],
        representation_pairs=primary['representation_pairs'],
        source_models=conditional.get('source_models', []),
        sid_variants=conditional.get('sid_variants', ['rqvae/coll']),
        uid_variants=conditional.get('uid_variants', ['flat']),
        hash_coders=conditional.get('hash_coders', ['simhash']),
        seeds=[int(seed) for seed in primary.get('seeds', ['42'])],
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
    parser.add_argument('--representations', '--pairs', default=None, help='Comma-separated pairs, e.g. uid2uid,sid2sid.')
    parser.add_argument('--seeds', default=None, help='Comma-separated random seeds, e.g. 42,43,44.')
    parser.add_argument('--targets', default=None, help='Legacy preselection; combined with --histories only.')
    parser.add_argument('--histories', default=None, help='Legacy preselection; combined with --targets only.')
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
