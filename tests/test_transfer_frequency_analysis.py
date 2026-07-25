import pandas as pd
import pytest

from scripts.transfer_frequency_analysis import pair_experiments, summarize_pair


def test_pairs_uid_and_sid_with_the_same_representation_addons():
    entries = [
        {
            'name': 'uid',
            'args': {
                'data': 'ras1',
                'model': 'scratch',
                'task_type': 'uid',
                'repr_type': 'uid+embedding',
                'repr_source_model': 'content',
            },
        },
        {
            'name': 'sid',
            'args': {
                'data': 'ras1',
                'model': 'scratch',
                'task_type': 'sid',
                'repr_type': 'sid+embedding',
                'repr_source_model': 'content',
            },
        },
        {
            'name': 'plain_sid',
            'args': {
                'data': 'ras1',
                'model': 'scratch',
                'task_type': 'sid',
                'repr_type': 'sid',
            },
        },
    ]

    pairs = pair_experiments(entries)

    assert len(pairs) == 1
    assert pairs[0][1]['name'] == 'uid'
    assert pairs[0][2]['name'] == 'sid'


def test_summarizes_sid_minus_uid_by_frequency_and_transfer_quality():
    uid = pd.DataFrame({
        'item_id': ['a', 'b', 'c', 'd'],
        'frequency': [2, 2, 30, 30],
        'frequency_bucket': ['1-5', '1-5', '21-100', '21-100'],
        'hr@10': [0.0, 0.0, 1.0, 1.0],
        'ndcg@10': [0.0, 0.0, 1.0, 1.0],
        'mrr': [0.0, 0.0, 1.0, 1.0],
    })
    sid = uid.copy()
    sid['hr@10'] = [0.0, 1.0, 0.0, 1.0]
    sid['ndcg@10'] = [0.0, 1.0, 0.0, 1.0]
    sid['mrr'] = [0.0, 1.0, 0.0, 1.0]
    transfer = pd.DataFrame({
        'item_id': ['a', 'b', 'c', 'd'],
        'content_ndcg@20': [0.1, 0.9, 0.2, 0.8],
    })

    _, summary = summarize_pair(
        uid,
        sid,
        transfer,
        tq_column='content_ndcg@20',
        k=10,
    )

    cold_high = summary[
        (summary['frequency_bucket'] == '1-5')
        & (summary['transfer_quality'] == 'high')
    ].iloc[0]
    warm_low = summary[
        (summary['frequency_bucket'] == '21-100')
        & (summary['transfer_quality'] == 'low')
    ].iloc[0]
    assert cold_high['delta_hr@10'] == pytest.approx(1.0)
    assert warm_low['delta_hr@10'] == pytest.approx(-1.0)
