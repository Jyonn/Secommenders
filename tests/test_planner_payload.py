from pathlib import Path

from planner import PlanSelections, _defaults, build_payload


def _selection(**overrides):
    values = {
        'name': 'smoke',
        'output': Path('/tmp/smoke.yaml'),
        'datasets': ['ras80'],
        'models': ['scratch'],
        'representation_pairs': ['uid2uid'],
        'source_models': [],
        'sid_variants': ['rqvae/coll'],
        'uid_variants': ['flat'],
        'hash_coders': ['simhash'],
        'seeds': [42],
        'params': _defaults(),
    }
    values.update(overrides)
    return PlanSelections(**values)


def test_uid_only_payload_drops_sid_and_decoder_args():
    params = _defaults()
    params['sid_codebook_size'] = 128
    params['code_beam_chunk_size'] = 80

    payload = build_payload(_selection(params=params))
    args = payload['experiments'][0]['args']

    assert args['task_type'] == 'uid'
    assert args['repr_type'] == 'uid'
    assert 'sid_codebook_size' not in args
    assert 'code_beam_chunk_size' not in args
    assert 'hash_num_bits' not in args


def test_text_pairs_skip_scratch_but_keep_llm_models():
    payload = build_payload(
        _selection(
            models=['scratch', 'qwen35th08b'],
            representation_pairs=['uid2uid', 'uid+text2uid'],
            seeds=[42, 43],
        )
    )
    names = [experiment['name'] for experiment in payload['experiments']]

    assert len(payload['experiments']) == 6
    assert any('scratch' in name and 'uid2uid' in name for name in names)
    assert not any('scratch' in name and 'uid+text2uid' in name for name in names)
    assert any('qwen35th08b' in name and 'uid+text2uid' in name for name in names)
    assert payload.get('planner_warnings')


def test_hash_payload_keeps_hash_params_only_when_hash_is_used():
    params = _defaults()
    params['hash_num_bits'] = 32

    payload = build_payload(
        _selection(
            representation_pairs=['hash2hash'],
            source_models=['pretrain-multimodal'],
            params=params,
        )
    )
    args = payload['experiments'][0]['args']

    assert args['task_type'] == 'hash'
    assert args['repr_type'] == 'hash'
    assert args['repr_source_model'] == 'pretrain-multimodal'
    assert args['hash_coder'] == 'simhash'
    assert args['hash_num_bits'] == 32
    assert 'sid_codebook_size' not in args
