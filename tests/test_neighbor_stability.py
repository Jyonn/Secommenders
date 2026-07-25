import pytest

from evaluator import rank_ndcg_scores


def test_rank_ndcg_is_one_for_identical_rankings():
    neighbors = {'anchor': ['a', 'b', 'c']}

    scores = rank_ndcg_scores(neighbors, neighbors, topk=3)

    assert scores == pytest.approx([1.0])


def test_rank_ndcg_penalizes_reversed_reference_order():
    reference = {'anchor': ['a', 'b', 'c']}
    reversed_target = {'anchor': ['c', 'b', 'a']}

    score = rank_ndcg_scores(reference, reversed_target, topk=3)[0]

    assert 0 < score < 1


def test_rank_ndcg_penalizes_missing_reference_neighbors():
    reference = {'anchor': ['a', 'b', 'c']}
    reversed_target = {'anchor': ['c', 'b', 'a']}
    missing_target = {'anchor': ['x', 'b', 'c']}

    reversed_score = rank_ndcg_scores(reference, reversed_target, topk=3)[0]
    missing_score = rank_ndcg_scores(reference, missing_target, topk=3)[0]

    assert missing_score < reversed_score


def test_rank_ndcg_skips_anchors_missing_from_target():
    reference = {'kept': ['a', 'b'], 'missing': ['a', 'b']}
    target = {'kept': ['a', 'b']}

    scores = rank_ndcg_scores(reference, target, topk=2)

    assert scores == pytest.approx([1.0])
