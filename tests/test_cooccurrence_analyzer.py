import random

import numpy as np
import pytest

from cooccurrence_analyzer import (
    add_quantization_loss_metrics,
    retrieval_metrics_from_scorer,
    sid_anchor_similarity,
)


def test_sid_anchor_similarity_uses_longest_common_prefix():
    codes = np.asarray(
        [
            [1, 2, 3, 4],
            [1, 2, 8, 9],
            [1, 7, 3, 4],
            [9, 2, 3, 4],
        ]
    )

    scores = sid_anchor_similarity(codes, 0)

    assert scores.tolist() == [4.0, 2.0, 1.0, 0.0]


def test_retrieval_uses_fixed_candidates_and_exports_per_item_metrics():
    scores = {
        0: np.asarray([10.0, 0.9, 1.0, 0.8], dtype=np.float32),
    }
    relevance = {0: {1: 2.0, 3: 1.0}}

    summary, per_item = retrieval_metrics_from_scorer(
        lambda anchor: scores[anchor],
        num_items=4,
        relevance=relevance,
        topks=[2],
        max_anchors=0,
        rng=random.Random(42),
        candidate_items={0, 1, 3},
        anchor_items=[0],
    )

    assert summary[0]['ndcg'] == pytest.approx(1.0)
    assert summary[0]['recall'] == pytest.approx(1.0)
    assert per_item[0]['ndcg@2'] == pytest.approx(1.0)


def test_retrieval_reuses_explicit_anchor_set():
    relevance = {
        0: {1: 1.0},
        1: {0: 1.0},
    }
    score_matrix = np.asarray(
        [
            [1.0, 0.9],
            [0.9, 1.0],
        ],
        dtype=np.float32,
    )

    summary, per_item = retrieval_metrics_from_scorer(
        lambda anchor: score_matrix[anchor],
        num_items=2,
        relevance=relevance,
        topks=[1],
        max_anchors=1,
        rng=random.Random(42),
        anchor_items=[1],
    )

    assert summary[0]['anchors'] == 1
    assert set(per_item) == {1}


def test_quantization_loss_is_content_minus_sid_ndcg():
    rows = {
        7: {
            'content_ndcg@20': 0.6,
            'sid_ndcg@20': 0.45,
        }
    }

    add_quantization_loss_metrics(rows, [20])

    assert rows[7]['quantization_loss_ndcg@20'] == pytest.approx(0.15)
