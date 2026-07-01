import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from utils.data import get_data_dir
from utils.logging import setup_logging


SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
DEFAULT_SYSTEM_PROMPT = '你是一位视频推荐系统专家，擅长捕捉用户的兴趣演变。请根据历史序列推荐后续视频。'
DEFAULT_USER_SUFFIX = '\n推荐后续视频：'


def _normalize_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [part for part in re.split(r'[\s,]+', text.strip('[]()')) if part]
        return _normalize_list(parsed)
    if pd.isna(value):
        return []
    return [value]


def _normalize_int_list(value) -> list[int]:
    result = []
    for item in _normalize_list(value):
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _parse_sid_triplet(value) -> tuple[int, int, int]:
    parts = _normalize_int_list(value)
    if len(parts) != 3:
        raise ValueError(f'Expected sid triplet of length 3, got {value!r}')
    return int(parts[0]), int(parts[1]), int(parts[2])


def _format_sid_sequence(pids: Iterable[int], sid_map: dict[int, tuple[int, int, int]]) -> str:
    tokens = []
    missing = []
    for pid in pids:
        triplet = sid_map.get(int(pid))
        if triplet is None:
            missing.append(int(pid))
            continue
        tokens.append(SID_FORMAT.format(c0=triplet[0], c1=triplet[1], c2=triplet[2]))
    if missing:
        raise ValueError(f'Missing SID mapping for {len(missing)} pid(s), first missing pid: {missing[0]}')
    return ''.join(tokens)


def _convert_sequence_to_pids(
        values,
        pid_by_row_index: list[int],
) -> list[int]:
    sequence = _normalize_int_list(values)
    invalid = [value for value in sequence if value < 0 or value >= len(pid_by_row_index)]
    if invalid:
        raise ValueError(
            f'Processed sequence contains {len(invalid)} out-of-range item row index value(s); '
            f'first invalid value: {invalid[0]}'
        )
    return [int(pid_by_row_index[value]) for value in sequence]


def _build_messages(history_sid_text: str, system_prompt: str, user_suffix: str) -> str:
    messages = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': system_prompt}],
        },
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': f'{history_sid_text}{user_suffix}'}],
        },
    ]
    return json.dumps(messages, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description='Export Secommenders processed RecIF video test split to an OpenOneRec benchmark-compatible parquet.'
    )
    parser.add_argument('--data', required=True, help='Dataset name, such as recifvideo or recifvideoxlarge.')
    parser.add_argument(
        '--processed_dir',
        default=None,
        help='Processed artifact directory. Defaults to artifacts/processed/<data>.',
    )
    parser.add_argument(
        '--sid_map',
        default=None,
        help='Path to video_ad_pid2sid.parquet. Defaults to <data_root>/video_ad_pid2sid.parquet.',
    )
    parser.add_argument(
        '--output_root',
        default=None,
        help='Benchmark-style output root. Defaults to artifacts/openonerec_eval/<data>.',
    )
    parser.add_argument(
        '--system_prompt',
        default=DEFAULT_SYSTEM_PROMPT,
        help='System prompt used in exported messages.',
    )
    parser.add_argument(
        '--user_suffix',
        default=DEFAULT_USER_SUFFIX,
        help='Suffix appended after the history SID sequence in the user message.',
    )
    args = parser.parse_args()

    data = str(args.data).lower()
    processed_dir = Path(args.processed_dir or f'artifacts/processed/{data}')
    if not processed_dir.exists():
        raise FileNotFoundError(f'Processed directory not found: {processed_dir}')

    data_root = get_data_dir(data)
    if args.sid_map is not None:
        sid_map_path = Path(args.sid_map)
    else:
        if not data_root:
            raise ValueError(
                f'Cannot resolve data root for {data}; please add it to .data or pass --sid_map explicitly.'
            )
        sid_map_path = Path(data_root) / 'video_ad_pid2sid.parquet'
    if not sid_map_path.exists():
        raise FileNotFoundError(f'SID mapping file not found: {sid_map_path}')

    output_root = Path(args.output_root or f'artifacts/openonerec_eval/{data}')
    output_path = output_root / 'video' / 'video_test.parquet'
    export_meta_path = output_root / 'video' / 'meta.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items_path = processed_dir / 'items.parquet'
    test_path = processed_dir / 'test.parquet'
    meta_path = processed_dir / 'meta.json'
    user_order_path = processed_dir / 'user_order.txt'
    for path in [items_path, test_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(f'Required processed artifact missing: {path}')

    pnt(f'loading processed items from {items_path}')
    items = pd.read_parquet(items_path)
    if 'pid' not in items.columns:
        raise ValueError(f'Expected processed items.parquet to contain a pid column, got {items.columns.tolist()}')

    pnt(f'loading processed test users from {test_path}')
    test_frame = pd.read_parquet(test_path)
    meta = json.loads(meta_path.read_text())
    history_col = str(meta.get('history_col', 'history'))
    uid_col = str(meta.get('user_col', 'uid'))
    answer_col = meta.get('multi_item_col')
    if history_col not in test_frame.columns:
        raise ValueError(f'Expected test.parquet to contain history column {history_col!r}')
    if uid_col not in test_frame.columns:
        raise ValueError(f'Expected test.parquet to contain uid column {uid_col!r}')

    pnt(f'loading SID mapping from {sid_map_path}')
    sid_frame = pd.read_parquet(sid_map_path)
    if 'pid' not in sid_frame.columns or 'sid' not in sid_frame.columns:
        raise ValueError(f'Expected sid mapping parquet to contain pid and sid columns, got {sid_frame.columns.tolist()}')
    sid_map = {
        int(row['pid']): _parse_sid_triplet(row['sid'])
        for _, row in sid_frame.iterrows()
    }

    pid_by_row_index = [int(pid) for pid in items['pid'].tolist()]
    pnt('processed test history id mode fixed to row-index -> items.parquet.pid')

    user_order_index = {}
    if user_order_path.exists():
        user_order_values = [line.strip() for line in user_order_path.read_text().splitlines() if line.strip()]
        user_order_index = {value: index for index, value in enumerate(user_order_values)}
        pnt(f'loaded user order index for {len(user_order_index)} users')

    records = []
    skipped_short = 0
    for _, row in tqdm(test_frame.iterrows(), total=len(test_frame), desc='export-video-test'):
        uid_value = row[uid_col]
        sequence_pids = _convert_sequence_to_pids(
            values=row[history_col],
            pid_by_row_index=pid_by_row_index,
        )

        if answer_col and answer_col in row and _normalize_int_list(row[answer_col]):
            history_pids = sequence_pids
            answer_pids = _convert_sequence_to_pids(
                values=row[answer_col],
                pid_by_row_index=pid_by_row_index,
            )
        else:
            if len(sequence_pids) < 2:
                skipped_short += 1
                continue
            history_pids = sequence_pids[:-1]
            answer_pids = [sequence_pids[-1]]

        history_sid_text = _format_sid_sequence(history_pids, sid_map)
        answer_sid_text = _format_sid_sequence(answer_pids, sid_map)

        metadata = {
            'answer': answer_sid_text,
            'uid': int(uid_value) if isinstance(uid_value, (int, np.integer)) else uid_value,
            'uuid': str(uuid.uuid4()),
            'answer_pid': [int(pid) for pid in answer_pids],
            'source_dataset': data,
        }
        user_order_key = str(uid_value)
        if user_order_key in user_order_index:
            metadata['user_order_index'] = int(user_order_index[user_order_key])

        records.append(
            {
                'hist_pid': np.asarray(history_pids, dtype=np.int64),
                'metadata': json.dumps(metadata, ensure_ascii=False),
                'messages': _build_messages(
                    history_sid_text=history_sid_text,
                    system_prompt=args.system_prompt,
                    user_suffix=args.user_suffix,
                ),
            }
        )

    if not records:
        raise ValueError(f'No exportable test samples were produced from {test_path}')

    output_frame = pd.DataFrame(records)
    output_frame.to_parquet(output_path, index=False)
    export_meta = {
        'source_dataset': data,
        'processed_dir': str(processed_dir),
        'items_path': str(items_path),
        'test_path': str(test_path),
        'sid_map_path': str(sid_map_path),
        'history_id_mode': 'row-index',
        'history_col': history_col,
        'uid_col': uid_col,
        'multi_item_col': answer_col,
        'system_prompt': args.system_prompt,
        'user_suffix': args.user_suffix,
        'sample_count': int(len(output_frame)),
        'skipped_short': int(skipped_short),
    }
    export_meta_path.write_text(json.dumps(export_meta, indent=2, ensure_ascii=False) + '\n')

    pnt(f'exported OpenOneRec-style video test file to {output_path}')
    pnt(
        f'samples={len(output_frame)} skipped_short={skipped_short} '
        f'items={len(items)} sid_map={len(sid_map)}'
    )


if __name__ == '__main__':
    setup_logging()
    main()
