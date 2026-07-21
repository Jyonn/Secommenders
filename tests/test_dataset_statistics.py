import json

import pandas as pd

from scripts.dataset_statistics import (
    SUMMARY_COLUMNS,
    SUMMARY_HEADERS,
    parse_datasets,
    render_table,
    summarize_frames,
    write_output,
)


def test_summarize_frames_computes_common_recommendation_statistics():
    items = pd.DataFrame({'iid': ['a', 'b', 'c', 'd']})
    users = pd.DataFrame(
        {
            'uid': ['u1', 'u2', 'u3'],
            'source_uid': ['s1', 's1', 's2'],
            'history': [['a', 'b'], ['a', 'c', 'a'], ['d']],
        }
    )
    test_users = pd.DataFrame({'uid': ['t1'], 'history': [['a', 'd']]})
    meta = {'item_col': 'iid', 'user_col': 'uid', 'history_col': 'history', 'scale_percent': 10}

    summary = summarize_frames('demo10', items, users, test_users, meta)

    assert summary['item_count'] == 4
    assert summary['observed_item_count'] == 4
    assert summary['user_count'] == 3
    assert summary['source_user_count'] == 2
    assert summary['interaction_count'] == 6
    assert summary['history_min'] == 1
    assert summary['history_median'] == 2
    assert summary['history_mean'] == 2
    assert summary['history_max'] == 3
    assert summary['density_percent'] == 50
    assert summary['sparsity_percent'] == 50
    assert summary['test_user_count'] == 1
    assert summary['test_interaction_count'] == 2


def test_parse_datasets_preserves_order_and_removes_duplicates():
    assert parse_datasets(' RAS1,ras2,ras1, ras5 ') == ['ras1', 'ras2', 'ras5']


def test_json_output_contains_full_record(tmp_path):
    output = tmp_path / 'stats.json'
    records = [{'data': 'ras1', 'item_count': 10, 'history_mean': 2.5}]
    write_output(output, records)
    assert json.loads(output.read_text()) == records


def test_summary_table_uses_compact_headers():
    record = {column: 1 for column in SUMMARY_COLUMNS}
    rendered = render_table([record], SUMMARY_COLUMNS, SUMMARY_HEADERS)
    header = rendered.splitlines()[0]

    assert '#Item' in header
    assert '#User' in header
    assert '#Inter' in header
    assert 'item_count' not in header
