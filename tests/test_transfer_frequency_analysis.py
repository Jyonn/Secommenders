import pandas as pd
import pytest

from scripts.transfer_frequency_analysis import (
    assign_transfer_quality,
    pair_experiments,
    summarize_pair,
)


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


def test_transfer_quality_is_split_independently_within_frequency_buckets():
    joined = pd.DataFrame({
        'item_id': list('abcdefgh'),
        'frequency_bucket': ['1-5'] * 4 + ['21-100'] * 4,
        'content_ndcg@20': [0.0, 0.1, 0.2, 0.3, 10.0, 11.0, 12.0, 13.0],
    })

    result = assign_transfer_quality(
        joined,
        tq_column='content_ndcg@20',
        tq_split='within-bucket',
    )

    counts = result.groupby(
        ['frequency_bucket', 'transfer_quality']
    ).size().to_dict()
    assert counts == {
        ('1-5', 'high'): 2,
        ('1-5', 'low'): 2,
        ('21-100', 'high'): 2,
        ('21-100', 'low'): 2,
    }
    thresholds = result.groupby('frequency_bucket')['tq_threshold'].first().to_dict()
    assert thresholds == {'1-5': 0.1, '21-100': 11.0}


def test_within_bucket_split_does_not_separate_tied_values():
    joined = pd.DataFrame({
        'item_id': list('abcdefg'),
        'frequency_bucket': ['1-5'] * 7,
        'content_ndcg@20': [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0],
    })

    result = assign_transfer_quality(
        joined,
        tq_column='content_ndcg@20',
        tq_split='within-bucket',
    )

    labels_per_value = result.groupby('content_ndcg@20')['transfer_quality'].nunique()
    assert labels_per_value.max() == 1
    assert set(result.loc[result['content_ndcg@20'] == 0.0, 'transfer_quality']) == {'low'}
    assert set(result.loc[result['content_ndcg@20'] >= 0.5, 'transfer_quality']) == {'high'}


def test_within_bucket_split_skips_bucket_without_tq_variation():
    joined = pd.DataFrame({
        'item_id': list('abcd'),
        'frequency_bucket': ['1-5'] * 4,
        'content_ndcg@20': [0.0] * 4,
    })

    result = assign_transfer_quality(
        joined,
        tq_column='content_ndcg@20',
        tq_split='within-bucket',
    )

    assert result.empty


def test_global_split_preserves_legacy_whole_pair_median():
    joined = pd.DataFrame({
        'item_id': list('abcdefgh'),
        'frequency_bucket': ['1-5'] * 4 + ['21-100'] * 4,
        'content_ndcg@20': [0.0, 0.1, 0.2, 0.3, 10.0, 11.0, 12.0, 13.0],
    })

    result = assign_transfer_quality(
        joined,
        tq_column='content_ndcg@20',
        tq_split='global',
    )

    assert result['tq_threshold'].nunique() == 1
    assert result['tq_threshold'].iloc[0] == pytest.approx(5.15)
    assert set(result.loc[result['frequency_bucket'] == '1-5', 'transfer_quality']) == {'low'}
    assert set(result.loc[result['frequency_bucket'] == '21-100', 'transfer_quality']) == {'high'}
