import numpy as np
import pytest

from candidate_analyzer import _common_prefix, _max_history_similarity


def test_common_prefix_respects_semantic_slot_limit():
    assert _common_prefix([1, 2, 3, 9], [1, 2, 4, 9]) == 2
    assert _common_prefix([1, 2, 3, 8], [1, 2, 3, 9], limit=3) == 3


def test_max_history_similarity_uses_closest_history_item():
    matrix = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6],
    ])
    assert _max_history_similarity(matrix, 2, [0, 1]) == pytest.approx(0.8)
