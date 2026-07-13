import json
import os
import hashlib
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


def _candidate_signature(candidate_pids: set | None):
    if candidate_pids is None:
        return None, None
    normalized = sorted(str(pid) for pid in candidate_pids)
    digest = hashlib.sha1()
    for pid in normalized:
        digest.update(pid.encode('utf-8'))
        digest.update(b'\0')
    return len(normalized), digest.hexdigest()


def _cache_matches(output_path: Path, candidate_pids: set | None) -> bool:
    if not output_path.exists():
        return False
    candidate_count, candidate_hash = _candidate_signature(candidate_pids)
    if candidate_hash is None:
        return True
    meta_path = output_path.with_name(FILTERED_EMBEDDINGS_META_NAME)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        meta.get('candidate_scope') == 'provided'
        and int(meta.get('candidate_count', -1)) == int(candidate_count)
        and meta.get('candidate_hash') == candidate_hash
    )


def _candidate_array(candidate_pids: set | None):
    if candidate_pids is None:
        return None
    return pa.array(list(candidate_pids))


def _filter_embedding_table(path: Path, candidate_values):
    table = pq.read_table(path, columns=list(REQUIRED_EMBEDDING_COLUMNS))
    if candidate_values is not None:
        try:
            candidate_mask = pc.is_in(table['pid'], value_set=candidate_values)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            candidate_mask = pc.is_in(
                table['pid'],
                value_set=pa.array(candidate_values.to_pylist(), type=table['pid'].type),
            )
        table = table.filter(candidate_mask)

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


def ensure_filtered_recif_embeddings(
    data_dir: str | Path,
    *,
    candidate_pids: set | None = None,
    workers: int | None = None,
    force: bool = False,
) -> Path:
    output_path = filtered_embeddings_path(data_dir)
    if not force and _cache_matches(output_path, candidate_pids):
        return output_path

    embeddings_dir = output_path.parent
    if not embeddings_dir.exists():
        raise FileNotFoundError(f'RecIF embedding directory not found: {embeddings_dir}')

    parquet_paths = raw_embedding_paths(data_dir)
    if not parquet_paths:
        raise FileNotFoundError(f'No raw RecIF embedding parquet files found in {embeddings_dir}')

    workers = max(1, int(workers or _default_workers()))
    candidate_count, candidate_hash = _candidate_signature(candidate_pids)
    candidate_values = _candidate_array(candidate_pids)
    temp_path = output_path.with_name(f'.{output_path.name}.{os.getpid()}.tmp')
    meta_path = output_path.with_name(FILTERED_EMBEDDINGS_META_NAME)
    if temp_path.exists():
        temp_path.unlink()

    pnt(
        f'building RecIF filtered embeddings cache files={len(parquet_paths)} '
        f'workers={workers} candidates={candidate_count if candidate_count is not None else "all"} '
        f'output={output_path}'
    )
    writer = None
    writer_schema = None
    matched_rows = 0
    kept_rows = 0
    completed_files = 0
    failed = False

    def submit_next(executor, pending, iterator):
        try:
            path = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(_filter_embedding_table, path, candidate_values))
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
                    matched_rows += int(total_rows)
                    kept_rows += int(valid_rows)
                    completed_files += 1
                    if filtered.num_rows:
                        if writer is None:
                            writer_schema = filtered.schema
                            writer = pq.ParquetWriter(temp_path, writer_schema, compression='snappy')
                        else:
                            filtered = filtered.cast(writer_schema)
                        writer.write_table(filtered)
                    progress.set_postfix(candidates=matched_rows, kept=kept_rows, invalid=matched_rows - kept_rows)
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
        'candidate_scope': 'provided' if candidate_hash is not None else 'all',
        'candidate_count': candidate_count,
        'candidate_hash': candidate_hash,
        'workers': workers,
        'columns': list(REQUIRED_EMBEDDING_COLUMNS),
        'matched_candidate_rows': int(matched_rows),
        'kept_rows': int(kept_rows),
        'invalid_candidate_rows': int(matched_rows - kept_rows),
        'complete_policy': 'pid must be non-null; vision_emb and text_emb must be non-null and non-empty',
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    pnt(
        f'RecIF filtered embeddings ready kept={kept_rows} '
        f'invalid={matched_rows - kept_rows} path={output_path}'
    )
    return output_path


def load_filtered_embedding_pids(
    data_dir: str | Path,
    *,
    candidate_pids: set | None = None,
    workers: int | None = None,
) -> set:
    path = ensure_filtered_recif_embeddings(data_dir, candidate_pids=candidate_pids, workers=workers)
    table = pq.read_table(path, columns=['pid'])
    return set(table.column('pid').to_pylist())
