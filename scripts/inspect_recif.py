import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def trim_value(value: Any, max_chars: int = 120):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        preview = value[:5]
        if len(value) > 5:
            return preview + ['...']
        return preview
    if isinstance(value, dict):
        keys = list(value)[:5]
        preview = {key: value[key] for key in keys}
        if len(value) > 5:
            preview['...'] = '...'
        return preview
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + '...'
    return value


def preview_records(df: pd.DataFrame, limit: int = 2):
    records = df.head(limit).to_dict(orient='records')
    return [
        {key: trim_value(value) for key, value in record.items()}
        for record in records
    ]


def list_like_columns(df: pd.DataFrame):
    candidates = []
    sample = df.head(min(len(df), 200))
    for column in df.columns:
        has_list = False
        avg_len = None
        for value in sample[column]:
            if isinstance(value, list):
                has_list = True
                break
        if not has_list:
            continue
        lengths = [len(value) for value in sample[column] if isinstance(value, list)]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
        candidates.append((column, avg_len))
    return candidates


def non_empty_mask(series: pd.Series):
    return series.apply(
        lambda value: (
            isinstance(value, (list, np.ndarray)) and len(value) > 0
        ) or (
            isinstance(value, str) and len(value.strip()) > 0
        )
    )


def summarize_sequence_columns(df: pd.DataFrame, columns: list[str]):
    print('sequence_stats=')
    for column in columns:
        if column not in df.columns:
            continue
        mask = non_empty_mask(df[column])
        present = int(mask.sum())
        lengths = df.loc[mask, column].apply(lambda value: len(value))
        avg_len = float(lengths.mean()) if present else 0.0
        max_len = int(lengths.max()) if present else 0
        print(f'  {column}: present={present}/{len(df)} avg_len={avg_len:.2f} max_len={max_len}')


def summarize_text_columns(df: pd.DataFrame, columns: list[str]):
    print('text_stats=')
    for column in columns:
        if column not in df.columns:
            continue
        mask = non_empty_mask(df[column])
        present = int(mask.sum())
        lengths = df.loc[mask, column].apply(lambda value: len(value))
        avg_len = float(lengths.mean()) if present else 0.0
        max_len = int(lengths.max()) if present else 0
        print(f'  {column}: present={present}/{len(df)} avg_chars={avg_len:.2f} max_chars={max_len}')


def unique_ids_from_sequence_column(series: pd.Series):
    values = set()
    for entry in series:
        if isinstance(entry, np.ndarray):
            entry = entry.tolist()
        if isinstance(entry, list):
            values.update(int(item) for item in entry)
    return values


def summarize_parquet(path: Path, head_rows: int = 2):
    print(f'\n=== FILE: {path}')
    df = pd.read_parquet(path)
    print(f'rows={len(df)} cols={len(df.columns)}')
    print('columns=', df.columns.tolist())
    print('dtypes=', {column: str(dtype) for column, dtype in df.dtypes.items()})
    list_columns = list_like_columns(df)
    if list_columns:
        print('list_like_columns=', {name: avg_len for name, avg_len in list_columns})
    print('head=', json.dumps(preview_records(df, head_rows), ensure_ascii=False, indent=2))

    candidate_columns = [
        column for column in df.columns
        if any(token in column.lower() for token in ['uid', 'user', 'sid', 'pid', 'item', 'hist', 'seq', 'time', 'domain', 'caption', 'label'])
    ]
    if candidate_columns:
        print('candidate_columns=', candidate_columns)
    return df


def summarize_json(path: Path, limit: int = 5):
    print(f'\n=== FILE: {path}')
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        print(f'type=dict size={len(data)}')
        sample_keys = list(data)[:limit]
        preview = {key: trim_value(data[key]) for key in sample_keys}
        print('sample=', json.dumps(preview, ensure_ascii=False, indent=2))
    elif isinstance(data, list):
        print(f'type=list size={len(data)}')
        print('sample=', json.dumps([trim_value(value) for value in data[:limit]], ensure_ascii=False, indent=2))
    else:
        print(f'type={type(data).__name__}')
        print(trim_value(data))


def main():
    root = Path.cwd()
    print(f'ROOT={root}')

    readme = root / 'README.md'
    if readme.exists():
        print('\n=== README HEAD')
        lines = readme.read_text(errors='ignore').splitlines()
        for line in lines[:80]:
            print(line)

    parquet_files = [
        root / 'onerec_bench_release.parquet',
        root / 'pid2caption.parquet',
        root / 'product_pid2sid.parquet',
        root / 'video_ad_pid2sid.parquet',
        root / 'benchmark_data' / 'product' / 'product_test.parquet',
        root / 'benchmark_data' / 'video' / 'video_test.parquet',
        root / 'benchmark_data' / 'ad' / 'ad_test.parquet',
        root / 'benchmark_data' / 'interactive' / 'interactive_test.parquet',
        root / 'benchmark_data' / 'item_understand' / 'item_understand_test.parquet',
        root / 'benchmark_data' / 'label_cond' / 'label_cond_test.parquet',
        root / 'benchmark_data' / 'label_pred' / 'label_pred_test.parquet',
        root / 'benchmark_data' / 'rec_reason' / 'rec_reason_test.parquet',
    ]

    loaded = {}
    for path in parquet_files:
        if path.exists():
            try:
                loaded[str(path)] = summarize_parquet(path)
            except Exception as exc:
                print(f'FAILED reading {path}: {exc}')

    for path in [root / 'benchmark_data' / 'sid2iid.json', root / 'benchmark_data' / 'sid2pid.json']:
        if path.exists():
            try:
                summarize_json(path)
            except Exception as exc:
                print(f'FAILED reading {path}: {exc}')

    main_path = str(root / 'onerec_bench_release.parquet')
    if main_path in loaded:
        df = loaded[main_path]
        print('\n=== HEURISTIC COLUMN CARDINALITIES (first 20 interesting columns)')
        interesting = [
            column for column in df.columns
            if any(token in column.lower() for token in ['uid', 'user', 'sid', 'pid', 'item', 'hist', 'seq', 'time', 'domain', 'label'])
        ][:20]
        for column in interesting:
            series = df[column]
            try:
                nunique = series.nunique(dropna=True)
            except Exception:
                nunique = 'n/a'
            print(f'{column}: nunique={nunique}')

        print('\n=== MAIN TABLE SPLIT STATS')
        if 'split' in df.columns:
            print('split_counts=', df['split'].value_counts(dropna=False).sort_index().to_dict())

        summarize_sequence_columns(
            df,
            [
                'hist_video_pid',
                'target_video_pid',
                'hist_ad_pid',
                'target_ad_pid',
                'hist_goods_pid',
                'target_goods_pid',
                'hist_longview_video_list',
            ],
        )
        summarize_text_columns(
            df,
            [
                'inter_keyword_to_items',
                'inter_user_profile_with_pid',
                'inter_user_profile_with_sid',
                'reco_gsu_caption',
                'reco_target_caption',
                'reco_cot',
            ],
        )

        print('\n=== UNIFIED HISTORY COVERAGE')
        history_columns = ['hist_video_pid', 'hist_ad_pid', 'hist_goods_pid', 'hist_longview_video_list']
        availability = {}
        for column in history_columns:
            if column in df.columns:
                availability[column] = int(non_empty_mask(df[column]).sum())
        print('history_presence=', availability)

        if all(column in df.columns for column in ['hist_video_pid', 'hist_ad_pid', 'hist_goods_pid']):
            any_primary_history = (
                non_empty_mask(df['hist_video_pid'])
                | non_empty_mask(df['hist_ad_pid'])
                | non_empty_mask(df['hist_goods_pid'])
            )
            print('users_with_any_primary_history=', int(any_primary_history.sum()))

        print('\n=== ITEM ID SPACE OVERLAP')
        video_ids = unique_ids_from_sequence_column(df['hist_video_pid']) | unique_ids_from_sequence_column(df['target_video_pid']) if 'hist_video_pid' in df.columns and 'target_video_pid' in df.columns else set()
        ad_ids = unique_ids_from_sequence_column(df['hist_ad_pid']) | unique_ids_from_sequence_column(df['target_ad_pid']) if 'hist_ad_pid' in df.columns and 'target_ad_pid' in df.columns else set()
        goods_ids = unique_ids_from_sequence_column(df['hist_goods_pid']) | unique_ids_from_sequence_column(df['target_goods_pid']) if 'hist_goods_pid' in df.columns and 'target_goods_pid' in df.columns else set()
        print('video_item_count=', len(video_ids))
        print('ad_item_count=', len(ad_ids))
        print('goods_item_count=', len(goods_ids))
        if video_ids and ad_ids:
            print('video_ad_overlap=', len(video_ids & ad_ids))
        if video_ids and goods_ids:
            print('video_goods_overlap=', len(video_ids & goods_ids))
        if ad_ids and goods_ids:
            print('ad_goods_overlap=', len(ad_ids & goods_ids))


if __name__ == '__main__':
    main()
