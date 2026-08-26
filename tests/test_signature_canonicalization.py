from copy import deepcopy

from utils.artifact_identity import (
    COMPILED_SPEC_VERSION,
    TRAIN_CONFIG_DEFAULTS,
    TRAINED_SPEC_VERSION,
    compiled_signature_from_config,
    migrate_train_config_dict,
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


def test_lr_scheduler_and_warmup_participate_in_trained_signature():
    baseline = _sid_train_config('llama3')
    cosine = deepcopy(baseline)
    cosine.update({'epochs': 20, 'lr_scheduler': 'cosine', 'warmup_ratio': 0.1})

    assert TRAINED_SPEC_VERSION == 'trained.v4'
    assert COMPILED_SPEC_VERSION == 'compiled.v3'
    assert trained_signature_from_config(baseline) != trained_signature_from_config(cosine)


def test_representation_pair_bias_only_changes_signature_when_enabled():
    baseline = _sid_train_config('llama3')
    explicit_disabled = deepcopy(baseline)
    explicit_disabled['representation_pair_bias'] = False
    enabled = deepcopy(baseline)
    enabled['representation_pair_bias'] = True

    assert trained_signature_from_config(baseline) == trained_signature_from_config(explicit_disabled)
    assert trained_signature_from_config(baseline) != trained_signature_from_config(enabled)

    shared_mode = deepcopy(baseline)
    shared_mode.update({
        'representation_pair_bias': True,
        'representation_pair_bias_mode': 'shared',
    })
    head_mode = deepcopy(baseline)
    head_mode['representation_pair_bias_mode'] = 'head'

    assert trained_signature_from_config(enabled) == trained_signature_from_config(shared_mode)
    assert trained_signature_from_config(enabled) != trained_signature_from_config(head_mode)


def test_legacy_train_config_migrates_with_historical_scheduler_defaults():
    legacy = _sid_train_config('llama3')
    legacy.pop('lr_scheduler', None)
    legacy.pop('warmup_ratio', None)

    migrated = migrate_train_config_dict(legacy)

    assert migrated['lr_scheduler'] == 'constant'
    assert migrated['warmup_ratio'] == 0.0
