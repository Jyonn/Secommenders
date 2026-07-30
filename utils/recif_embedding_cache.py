import json
import os
import hashlib
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import fcntl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np
from pigmento import pnt
from tqdm import tqdm


FILTERED_EMBEDDINGS_NAME = 'filtered.parquet'
FILTERED_EMBEDDINGS_META_NAME = 'filtered.meta.json'
FILTERED_EMBEDDINGS_VERSION = 'v2-mean-pooled'
REQUIRED_EMBEDDING_COLUMNS = ('pid', 'vision_emb', 'text_emb')
SCALE_DATASET_PATTERN = re.compile(r'^(ras|rvs|rvt|ra|rv)\d+$')


def recif_embeddings_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / 'embeddings'


def recif_dataset_root(dataset: str) -> str:
    normalized = str(dataset).strip().lower()
    match = SCALE_DATASET_PATTERN.fullmatch(normalized)
    return match.group(1) if match else normalized


def filtered_embeddings_path(
    data_dir: str | Path,
    *,
    dataset: str | None = None,
    candidate_pids: set | None = None,
) -> Path:
    embeddings_dir = recif_embeddings_dir(data_dir)
    if dataset:
        return embeddings_dir / f'filtered.{recif_dataset_root(dataset)}.parquet'
    candidate_count, candidate_hash = _candidate_signature(candidate_pids)
    if candidate_hash is None:
        return embeddings_dir / FILTERED_EMBEDDINGS_NAME
    return embeddings_dir / f'filtered.{candidate_count}.{candidate_hash[:16]}.parquet'


def filtered_embeddings_meta_path(output_path: Path) -> Path:
    if output_path.name == FILTERED_EMBEDDINGS_NAME:
        return output_path.with_name(FILTERED_EMBEDDINGS_META_NAME)
    return output_path.with_suffix('.meta.json')


def filtered_embeddings_candidates_path(output_path: Path) -> Path:
    return output_path.with_suffix('.candidates.parquet')


def raw_embedding_paths(data_dir: str | Path) -> list[Path]:
    embeddings_dir = recif_embeddings_dir(data_dir)
    return [
        path
        for path in sorted(embeddings_dir.glob('*.parquet'))
        if not path.name.startswith('filtered.')
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


def _load_meta(output_path: Path) -> dict | None:
    meta_path = filtered_embeddings_meta_path(output_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None


def _load_cached_pids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    return set(pq.read_table(output_path, columns=['pid']).column('pid').to_pylist())


def _pid_keys(pids) -> set[str]:
    return {str(pid) for pid in pids}


def _load_covered_pid_keys(output_path: Path) -> set[str]:
    candidates_path = filtered_embeddings_candidates_path(output_path)
    if not candidates_path.exists():
        return set()
    table = pq.read_table(candidates_path, columns=['pid'])
    return set(table.column('pid').to_pylist())


def _cache_matches(
    output_path: Path,
    candidate_pids: set | None,
    *,
    cache_root: str | None = None,
) -> bool:
    if not output_path.exists():
        return False
    candidate_count, candidate_hash = _candidate_signature(candidate_pids)
    meta = _load_meta(output_path)
    if meta is None or meta.get('version') != FILTERED_EMBEDDINGS_VERSION:
        return False
    if cache_root is not None:
        if meta.get('cache_root') != cache_root:
            return False
        if candidate_pids is None:
            return meta.get('candidate_scope') == 'all'
        covered_pid_keys = _load_covered_pid_keys(output_path)
        return bool(covered_pid_keys) and _pid_keys(candidate_pids).issubset(covered_pid_keys)
    if candidate_hash is None:
        return meta.get('candidate_scope') == 'all'
    return (
        meta.get('candidate_scope') == 'provided'
        and int(meta.get('candidate_count', -1)) == int(candidate_count)
        and meta.get('candidate_hash') == candidate_hash
    )


@contextmanager
def _cache_lock(output_path: Path):
    lock_path = output_path.with_suffix(f'{output_path.suffix}.lock')
    with lock_path.open('a+') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _candidate_array(candidate_pids: set | None):
    if candidate_pids is None:
        return None
    return pa.array(list(candidate_pids))


def _as_vector(value):
    if value is None:
        raise ValueError('embedding value is null')
    if hasattr(value, 'as_py'):
        value = value.as_py()
    if isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 1:
        vectors = [_as_vector(item) for item in value.tolist() if item is not None]
        if not vectors:
            raise ValueError('embedding value is empty')
        dims = {int(vector.shape[0]) for vector in vectors}
        if len(dims) == 1 and len(vectors) > 1:
            return np.stack(vectors, axis=0).mean(axis=0).astype(np.float32)
        if len(vectors) == 1:
            return vectors[0]
    if isinstance(value, (list, tuple)) and value and any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
        vectors = [_as_vector(item) for item in value if item is not None]
        if not vectors:
            raise ValueError('embedding value is empty')
        dims = {int(vector.shape[0]) for vector in vectors}
        if len(dims) == 1 and len(vectors) > 1:
            return np.stack(vectors, axis=0).mean(axis=0).astype(np.float32)
        if len(vectors) == 1:
            return vectors[0]

    def flatten(item):
        if hasattr(item, 'as_py'):
            item = item.as_py()
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, (list, tuple)):
            for child in item:
                yield from flatten(child)
            return
        if item is None:
            return
        yield float(item)

    vector = np.fromiter(flatten(value), dtype=np.float32)
    if vector.size == 0:
        raise ValueError('embedding value is empty')
    return vector


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
    rows = []
    for row in filtered.to_pylist():
        try:
            rows.append(
                {
                    'pid': row['pid'],
                    'vision_emb': _as_vector(row['vision_emb']).tolist(),
                    'text_emb': _as_vector(row['text_emb']).tolist(),
                }
            )
        except ValueError:
            continue
    pooled = pa.Table.from_pylist(rows, schema=pa.schema([
        ('pid', table['pid'].type),
        ('vision_emb', pa.list_(pa.float32())),
        ('text_emb', pa.list_(pa.float32())),
    ]))
    return path, table.num_rows, pooled.num_rows, pooled


def _build_filtered_recif_embeddings(
    data_dir: str | Path,
    *,
    output_path: Path,
    candidate_pids: set | None = None,
    covered_pid_keys: set[str] | None = None,
    workers: int | None = None,
    cache_root: str | None = None,
) -> Path:
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
    meta_path = filtered_embeddings_meta_path(output_path)
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
    if candidate_pids is not None:
        coverage_keys = covered_pid_keys or _pid_keys(candidate_pids)
        candidates_path = filtered_embeddings_candidates_path(output_path)
        candidates_temp_path = candidates_path.with_name(f'.{candidates_path.name}.{os.getpid()}.tmp')
        coverage_table = pa.table({'pid': sorted(coverage_keys)})
        pq.write_table(coverage_table, candidates_temp_path, compression='snappy')
        candidates_temp_path.replace(candidates_path)
    else:
        coverage_keys = None
    meta = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'version': FILTERED_EMBEDDINGS_VERSION,
        'cache_root': cache_root,
        'source_dir': str(embeddings_dir),
        'source_files': len(parquet_paths),
        'completed_files': int(completed_files),
        'candidate_scope': 'provided' if candidate_hash is not None else 'all',
        'candidate_count': len(coverage_keys) if coverage_keys is not None else candidate_count,
        'candidate_hash': (
            _candidate_signature(coverage_keys)[1] if coverage_keys is not None else candidate_hash
        ),
        'candidate_coverage_path': (
            str(filtered_embeddings_candidates_path(output_path))
            if coverage_keys is not None
            else None
        ),
        'workers': workers,
        'columns': list(REQUIRED_EMBEDDING_COLUMNS),
        'matched_candidate_rows': int(matched_rows),
        'kept_rows': int(kept_rows),
        'invalid_candidate_rows': int(matched_rows - kept_rows),
        'complete_policy': 'pid must be non-null; vision_emb and text_emb must be non-null and non-empty',
        'storage_policy': 'vision_emb is mean-pooled to one vector; text_emb is stored as one vector',
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    pnt(
        f'RecIF filtered embeddings ready kept={kept_rows} '
        f'invalid={matched_rows - kept_rows} path={output_path}'
    )
    return output_path


def ensure_filtered_recif_embeddings(
    data_dir: str | Path,
    *,
    dataset: str | None = None,
    candidate_pids: set | None = None,
    workers: int | None = None,
    force: bool = False,
) -> Path:
    cache_root = recif_dataset_root(dataset) if dataset else None
    output_path = filtered_embeddings_path(
        data_dir,
        dataset=dataset,
        candidate_pids=candidate_pids,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _cache_lock(output_path):
        if not force and _cache_matches(output_path, candidate_pids, cache_root=cache_root):
            return output_path

        build_candidates = candidate_pids
        covered_pid_keys = _pid_keys(candidate_pids) if candidate_pids is not None else None
        meta = _load_meta(output_path)
        if (
            cache_root is not None
            and candidate_pids is not None
            and meta is not None
            and meta.get('version') == FILTERED_EMBEDDINGS_VERSION
            and meta.get('cache_root') == cache_root
        ):
            cached_pids = _load_cached_pids(output_path)
            cached_coverage = _load_covered_pid_keys(output_path)
            build_candidates = set(candidate_pids) | cached_pids
            covered_pid_keys = cached_coverage | _pid_keys(candidate_pids)
            missing_count = len(_pid_keys(candidate_pids) - cached_coverage)
            if missing_count:
                pnt(
                    f'expanding RecIF filtered embedding cache root={cache_root} '
                    f'cached={len(cached_pids)} requested={len(candidate_pids)} missing={missing_count}'
                )

        return _build_filtered_recif_embeddings(
            data_dir,
            output_path=output_path,
            candidate_pids=build_candidates,
            covered_pid_keys=covered_pid_keys,
            workers=workers,
            cache_root=cache_root,
        )


def load_filtered_embedding_pids(
    data_dir: str | Path,
    *,
    dataset: str | None = None,
    candidate_pids: set | None = None,
    workers: int | None = None,
) -> set:
    path = ensure_filtered_recif_embeddings(
        data_dir,
        dataset=dataset,
        candidate_pids=candidate_pids,
        workers=workers,
    )
    table = pq.read_table(path, columns=['pid'])
    return set(table.column('pid').to_pylist())
