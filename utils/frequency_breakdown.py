import math
from collections import Counter

import numpy as np


DEFAULT_FREQUENCY_BOUNDARIES = [0, 5, 20, 100]


def normalize_frequency_boundaries(values):
    if values is None:
        return list(DEFAULT_FREQUENCY_BOUNDARIES)
    if isinstance(values, str):
        values = [part.strip() for part in values.split(',') if part.strip()]
    boundaries = sorted({int(value) for value in values})
    if not boundaries or boundaries[0] != 0:
        raise ValueError('frequency buckets must start with 0')
    if any(value < 0 for value in boundaries):
        raise ValueError('frequency bucket boundaries must be non-negative')
    return boundaries


def count_finetune_target_frequencies(finetune_frame):
    """Count item occurrences at positions supervised by next-item prediction."""
    frequencies = Counter()
    for sequence in finetune_frame['sequence_uids'].tolist():
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        elif isinstance(sequence, tuple):
            sequence = list(sequence)
        sequence = [int(uid) for uid in sequence]
        frequencies.update(sequence[1:])
    return frequencies


def frequency_bucket(frequency: int, boundaries):
    frequency = int(frequency)
    boundaries = normalize_frequency_boundaries(boundaries)
    if frequency == 0:
        return '0'
    lower = 1
    for upper in boundaries[1:]:
        if frequency <= upper:
            return f'{lower}-{upper}'
        lower = upper + 1
    return f'{lower}+'


class FrequencyBreakdownAccumulator:
    def __init__(self, frequencies, boundaries, ks):
        self.frequencies = Counter({int(uid): int(value) for uid, value in frequencies.items()})
        self.boundaries = normalize_frequency_boundaries(boundaries)
        self.ks = sorted({int(k) for k in ks})
        self._buckets = {}
        self._records = []

    def _bucket(self, label):
        if label not in self._buckets:
            self._buckets[label] = {
                'target_count': 0,
                'target_uids': set(),
                'frequencies': [],
                'mrr_sum': 0.0,
                'hr_sums': {k: 0.0 for k in self.ks},
                'ndcg_sums': {k: 0.0 for k in self.ks},
            }
        return self._buckets[label]

    def add(self, target_uid: int, rank: int | None, raw_item_id=None):
        target_uid = int(target_uid)
        frequency = int(self.frequencies.get(target_uid, 0))
        label = frequency_bucket(frequency, self.boundaries)
        self._records.append({
            'target_uid': target_uid,
            'raw_item_id': str(raw_item_id) if raw_item_id is not None else None,
            'frequency': frequency,
            'frequency_bucket': label,
            'rank': int(rank) if rank is not None else None,
        })
        bucket = self._bucket(label)
        bucket['target_count'] += 1
        bucket['target_uids'].add(target_uid)
        bucket['frequencies'].append(frequency)
        if rank is None:
            return
        rank = int(rank)
        bucket['mrr_sum'] += 1.0 / rank
        for k in self.ks:
            if rank <= k:
                bucket['hr_sums'][k] += 1.0
                bucket['ndcg_sums'][k] += 1.0 / math.log2(rank + 1)

    def summary(self):
        labels = ['0']
        lower = 1
        for upper in self.boundaries[1:]:
            labels.append(f'{lower}-{upper}')
            lower = upper + 1
        labels.append(f'{lower}+')

        total_targets = sum(bucket['target_count'] for bucket in self._buckets.values())
        buckets = {}
        for label in labels:
            bucket = self._buckets.get(label)
            if bucket is None:
                count = 0
                target_uids = set()
                frequencies = []
                mrr_sum = 0.0
                hr_sums = {k: 0.0 for k in self.ks}
                ndcg_sums = {k: 0.0 for k in self.ks}
            else:
                count = int(bucket['target_count'])
                target_uids = bucket['target_uids']
                frequencies = bucket['frequencies']
                mrr_sum = float(bucket['mrr_sum'])
                hr_sums = bucket['hr_sums']
                ndcg_sums = bucket['ndcg_sums']
            frequency_array = np.asarray(frequencies, dtype=np.float64)
            metrics = {'mrr': mrr_sum / count if count else None}
            for k in self.ks:
                metrics[f'hr@{k}'] = hr_sums[k] / count if count else None
                metrics[f'ndcg@{k}'] = ndcg_sums[k] / count if count else None
            buckets[label] = {
                'target_count': count,
                'target_share': count / total_targets if total_targets else None,
                'unique_item_count': len(target_uids),
                'frequency_mean': float(frequency_array.mean()) if count else None,
                'frequency_median': float(np.median(frequency_array)) if count else None,
                **metrics,
            }
        return {
            'frequency_definition': 'compiled_finetune_target_positions',
            'boundaries': list(self.boundaries),
            'total_test_targets': total_targets,
            'buckets': buckets,
            'records': list(self._records),
        }
