import json
from pathlib import Path
from typing import Any

import pandas as pd


def trim_value(value: Any, max_chars: int = 120):
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


if __name__ == '__main__':
    main()
