import math

import pandas as pd
import pytest

from utils.frequency_breakdown import (
    FrequencyBreakdownAccumulator,
    count_finetune_target_frequencies,
    frequency_bucket,
)


def test_counts_only_supervised_finetune_positions():
    frame = pd.DataFrame(
        {
            'sequence_uids': [
                [1, 2, 3],
                [1, 2, 2],
            ]
        }
    )

    frequencies = count_finetune_target_frequencies(frame)

    assert frequencies == {2: 3, 3: 1}
    assert frequencies[1] == 0


def test_frequency_buckets_use_fixed_absolute_boundaries():
    boundaries = [0, 5, 20, 100]

    assert frequency_bucket(0, boundaries) == '0'
    assert frequency_bucket(1, boundaries) == '1-5'
    assert frequency_bucket(5, boundaries) == '1-5'
    assert frequency_bucket(6, boundaries) == '6-20'
    assert frequency_bucket(20, boundaries) == '6-20'
    assert frequency_bucket(21, boundaries) == '21-100'
    assert frequency_bucket(101, boundaries) == '101+'


def test_breakdown_aggregates_single_target_ranking_metrics():
    accumulator = FrequencyBreakdownAccumulator(
        frequencies={10: 2, 20: 8},
        boundaries=[0, 5, 20, 100],
        ks=[5, 10],
    )
    accumulator.add(target_uid=10, rank=1)
    accumulator.add(target_uid=10, rank=None)
    accumulator.add(target_uid=20, rank=10)

    summary = accumulator.summary()

    extreme_cold = summary['buckets']['1-5']
    assert extreme_cold['target_count'] == 2
    assert extreme_cold['unique_item_count'] == 1
    assert extreme_cold['target_share'] == pytest.approx(2 / 3)
    assert extreme_cold['hr@5'] == pytest.approx(0.5)
    assert extreme_cold['ndcg@5'] == pytest.approx(0.5)
    assert extreme_cold['mrr'] == pytest.approx(0.5)

    cold = summary['buckets']['6-20']
    assert cold['target_count'] == 1
    assert cold['hr@5'] == 0
    assert cold['hr@10'] == 1
    assert cold['ndcg@10'] == pytest.approx(1 / math.log2(11))
    assert summary['records'][0] == {
        'target_uid': 10,
        'raw_item_id': None,
        'frequency': 2,
        'frequency_bucket': '1-5',
        'rank': 1,
    }


def test_breakdown_records_raw_item_id_for_transfer_quality_join():
    accumulator = FrequencyBreakdownAccumulator(
        frequencies={3: 7},
        boundaries=[0, 5, 20, 100],
        ks=[10],
    )

    accumulator.add(target_uid=3, rank=4, raw_item_id='news-42')

    assert accumulator.summary()['records'] == [{
        'target_uid': 3,
        'raw_item_id': 'news-42',
        'frequency': 7,
        'frequency_bucket': '6-20',
        'rank': 4,
    }]


def test_empty_buckets_are_preserved_with_null_metrics():
    accumulator = FrequencyBreakdownAccumulator(
        frequencies={10: 2},
        boundaries=[0, 5, 20, 100],
        ks=[10],
    )
    accumulator.add(target_uid=10, rank=1)

    summary = accumulator.summary()

    assert summary['buckets']['0']['target_count'] == 0
    assert summary['buckets']['0']['hr@10'] is None
    assert summary['buckets']['101+']['target_count'] == 0
