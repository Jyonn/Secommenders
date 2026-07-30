import json

import pyarrow as pa
import pyarrow.parquet as pq

from utils.recif_embedding_cache import (
    FILTERED_EMBEDDINGS_VERSION,
    _cache_matches,
    filtered_embeddings_candidates_path,
    filtered_embeddings_meta_path,
    filtered_embeddings_path,
    recif_dataset_root,
)


def test_recif_scale_datasets_share_family_cache_roots(tmp_path):
    assert recif_dataset_root('ras10') == 'ras'
    assert recif_dataset_root('ras99') == 'ras'
    assert recif_dataset_root('rvs20') == 'rvs'
    assert recif_dataset_root('rvt95') == 'rvt'
    assert recif_dataset_root('ra80') == 'ra'
    assert recif_dataset_root('rv90') == 'rv'

    assert filtered_embeddings_path(tmp_path, dataset='ras10').name == 'filtered.ras.parquet'
    assert filtered_embeddings_path(tmp_path, dataset='ras99').name == 'filtered.ras.parquet'


def test_recif_full_datasets_keep_independent_cache_roots(tmp_path):
    assert recif_dataset_root('raf') == 'raf'
    assert recif_dataset_root('rvf') == 'rvf'
    assert filtered_embeddings_path(tmp_path, dataset='raf').name == 'filtered.raf.parquet'
    assert filtered_embeddings_path(tmp_path, dataset='rvf').name == 'filtered.rvf.parquet'


def test_family_cache_tracks_checked_candidates_separately_from_valid_rows(tmp_path):
    output_path = filtered_embeddings_path(tmp_path, dataset='ras10')
    output_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'pid': [1]}), output_path)
    pq.write_table(
        pa.table({'pid': ['1', '2']}),
        filtered_embeddings_candidates_path(output_path),
    )
    filtered_embeddings_meta_path(output_path).write_text(
        json.dumps(
            {
                'version': FILTERED_EMBEDDINGS_VERSION,
                'cache_root': 'ras',
                'candidate_scope': 'provided',
            }
        )
    )

    assert _cache_matches(output_path, {1}, cache_root='ras')
    assert _cache_matches(output_path, {2}, cache_root='ras')
    assert not _cache_matches(output_path, {3}, cache_root='ras')
