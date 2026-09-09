from types import SimpleNamespace

import numpy as np
import pytest

from candidate_analyzer import _aggregate, _common_prefix, _expand_runtime_topk, _max_history_similarity


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


def test_runtime_topk_expands_sid_and_fusion_without_rebuilding_config():
    target = {'representation': 'sid_content', 'decoding': {'beam_width': 20}}
    compile_config = SimpleNamespace(
        representation_graph={'decoder': {'targets': [target]}},
        representation_kind=lambda name: 'sid',
    )
    config = SimpleNamespace(
        code_beam_width=20,
        multi_candidate_topk=100,
        multi_output_topk=20,
        compile_config=compile_config,
    )

    _expand_runtime_topk(config, 100)

    assert config.code_beam_width == 100
    assert config.multi_candidate_topk == 100
    assert config.multi_output_topk == 100
    assert target['decoding']['beam_width'] == 100


def test_aggregate_counts_strict_fusion_rank_wins():
    cases = [{
        'topk_overlap_jaccard': 0.2,
        'target_recalled_by_uid': True,
        'target_recalled_by_sid': True,
        'target_recalled_by_fused': True,
        'fusion_rank_win': True,
        'candidates': [],
    }]
    summary = _aggregate(cases)
    assert summary['fusion_rank_win_count'] == 1
    assert summary['fusion_rank_win_rate'] == 1.0
