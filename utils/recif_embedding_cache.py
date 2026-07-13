import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pigmento import pnt
from tqdm import tqdm


FILTERED_EMBEDDINGS_NAME = 'filtered.parquet'
FILTERED_EMBEDDINGS_META_NAME = 'filtered.meta.json'
REQUIRED_EMBEDDING_COLUMNS = ('pid', 'vision_emb', 'text_emb')


def recif_embeddings_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / 'embeddings'


def filtered_embeddings_path(data_dir: str | Path) -> Path:
    return recif_embeddings_dir(data_dir) / FILTERED_EMBEDDINGS_NAME


def raw_embedding_paths(data_dir: str | Path) -> list[Path]:
    embeddings_dir = recif_embeddings_dir(data_dir)
    return [
        path
        for path in sorted(embeddings_dir.glob('*.parquet'))
        if path.name != FILTERED_EMBEDDINGS_NAME
    ]


def _default_workers() -> int:
    raw_value = os.environ.get('RECIF_EMBEDDING_FILTER_WORKERS')
    if raw_value:
        return max(1, int(raw_value))
    return min(8, max(1, os.cpu_count() or 1))


def _filter_embedding_table(path: Path):
    table = pq.read_table(path, columns=list(REQUIRED_EMBEDDING_COLUMNS))
    vision_mask = pc.is_valid(table['vision_emb'])
    text_mask = pc.is_valid(table['text_emb'])
    try:
        vision_mask = pc.and_(vision_mask, pc.greater(pc.list_value_length(table['vision_emb']), 0))
        text_mask = pc.and_(text_mask, pc.greater(pc.list_value_length(table['text_emb']), 0))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
        pass
    mask = pc.and_(
        pc.and_(pc.is_valid(table['pid']), vision_mask),
        text_mask,
    )
    filtered = table.filter(mask)
    return path, table.num_rows, filtered.num_rows, filtered


def ensure_filtered_recif_embeddings(data_dir: str | Path, *, workers: int | None = None, force: bool = False) -> Path:
    output_path = filtered_embeddings_path(data_dir)
    if output_path.exists() and not force:
        return output_path

    embeddings_dir = output_path.parent
    if not embeddings_dir.exists():
        raise FileNotFoundError(f'RecIF embedding directory not found: {embeddings_dir}')

    parquet_paths = raw_embedding_paths(data_dir)
    if not parquet_paths:
        raise FileNotFoundError(f'No raw RecIF embedding parquet files found in {embeddings_dir}')

    workers = max(1, int(workers or _default_workers()))
    temp_path = output_path.with_name(f'.{output_path.name}.{os.getpid()}.tmp')
    meta_path = output_path.with_name(FILTERED_EMBEDDINGS_META_NAME)
    if temp_path.exists():
        temp_path.unlink()

    pnt(
        f'building RecIF filtered embeddings cache files={len(parquet_paths)} '
        f'workers={workers} output={output_path}'
    )
    writer = None
    writer_schema = None
    scanned_rows = 0
    kept_rows = 0
    completed_files = 0
    failed = False

    def submit_next(executor, pending, iterator):
        try:
            path = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(_filter_embedding_table, path))
        return True

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            path_iter = iter(parquet_paths)
            pending = set()
            for _ in range(min(workers * 2, len(parquet_paths))):
                submit_next(executor, pending, path_iter)

            progress = tqdm(total=len(parquet_paths), desc='filter-embeddings')
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    path, total_rows, valid_rows, filtered = future.result()
                    scanned_rows += int(total_rows)
                    kept_rows += int(valid_rows)
                    completed_files += 1
                    if filtered.num_rows:
                        if writer is None:
                            writer_schema = filtered.schema
                            writer = pq.ParquetWriter(temp_path, writer_schema, compression='snappy')
                        else:
                            filtered = filtered.cast(writer_schema)
                        writer.write_table(filtered)
                    progress.set_postfix(kept=kept_rows, dropped=scanned_rows - kept_rows)
                    progress.update(1)
                    submit_next(executor, pending, path_iter)
            progress.close()
    except Exception:
        failed = True
        raise
    finally:
        if writer is not None:
            writer.close()
        if failed and temp_path.exists():
            temp_path.unlink()

    if writer is None:
        raise ValueError(f'No RecIF embedding rows with complete vision/text embeddings found in {embeddings_dir}')

    temp_path.replace(output_path)
    meta = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_dir': str(embeddings_dir),
        'source_files': len(parquet_paths),
        'completed_files': int(completed_files),
        'workers': workers,
        'columns': list(REQUIRED_EMBEDDING_COLUMNS),
        'scanned_rows': int(scanned_rows),
        'kept_rows': int(kept_rows),
        'dropped_rows': int(scanned_rows - kept_rows),
        'complete_policy': 'pid must be non-null; vision_emb and text_emb must be non-null and non-empty',
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    pnt(
        f'RecIF filtered embeddings ready kept={kept_rows} '
        f'dropped={scanned_rows - kept_rows} path={output_path}'
    )
    return output_path


def load_filtered_embedding_pids(data_dir: str | Path, *, workers: int | None = None) -> set:
    path = ensure_filtered_recif_embeddings(data_dir, workers=workers)
    table = pq.read_table(path, columns=['pid'])
    return set(table.column('pid').to_pylist())
