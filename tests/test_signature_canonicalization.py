from copy import deepcopy

from utils.artifact_identity import (
    TRAIN_CONFIG_DEFAULTS,
    compiled_signature_from_config,
    trained_signature_from_config,
)
from utils.compile import CompileConfig
from utils.experiment_template import build_default_upstreams


def _sid_train_config(sid_embedding_model):
    config = deepcopy(TRAIN_CONFIG_DEFAULTS)
    config.update(
        {
            'data': 'mind',
            'model': 'scratch',
            'repr_type': 'sid',
            'task_type': 'sid',
            'repr_source_model': 'llama3',
            'sid_coder': 'rqvae',
            'sid_export': 'coll',
        }
    )
    config['upstreams'] = build_default_upstreams(
        {
            **config,
            'sid_embedding_model': sid_embedding_model,
        }
    )
    return config


def test_inherited_and_explicit_sid_embedding_models_share_signature():
    inherited = _sid_train_config(None)
    explicit = _sid_train_config('llama3')

    assert inherited['upstreams']['sid']['embedding_model'] == 'llama3'
    assert inherited['upstreams'] == explicit['upstreams']
    inherited_compile = CompileConfig(
        data='mind',
        model='scratch',
        repr_type='sid',
        repr_source_model='llama3',
        sid_export='coll',
        sid_coder='rqvae',
        hash_coder=None,
        task_type='sid',
        maxitems=50,
        model_max_length=None,
        item_text_max_tokens=20,
        repr_combine='concat',
        upstreams=inherited['upstreams'],
    )
    explicit_compile = CompileConfig(
        **{
            **inherited_compile.__dict__,
            'upstreams': explicit['upstreams'],
        }
    )
    assert compiled_signature_from_config(inherited_compile) == compiled_signature_from_config(explicit_compile)
    assert trained_signature_from_config(inherited) == trained_signature_from_config(explicit)
