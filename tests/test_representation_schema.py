from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import torch
import yaml
from oba import Obj

from core.train_config import TrainConfig
from core.model import SequentialRecModel
from trainer import Trainer
from utils.artifact_identity import trained_signature_from_config
from utils.config_init import ConfigInit
from utils.representation_schema import normalize_representation_graph, semantic_graph_contract


def _catalog():
    return {
        'uid': {'type': 'uid'},
        'sid_hybrid': {
            'type': 'sid',
            'sources': [{'model': 'llama3'}, {'model': 'word2vec', 'config': {'vector_size': 64}}],
            'codec': {'name': 'rqvae', 'export': 'recon', 'quantizer': {'codebook_size': 128}},
        },
        'embedding_llama': {'type': 'embedding', 'sources': [{'model': 'llama3'}]},
        'embedding_collab': {'type': 'embedding', 'sources': [{'model': 'word2vec'}]},
        'unused_sid': {'type': 'sid', 'sources': [{'model': 'qwen'}]},
    }


def test_graph_prunes_unreferenced_declarations_and_keeps_multiple_embeddings():
    graph = normalize_representation_graph(
        _catalog(),
        {'representations': ['uid', 'sid_hybrid', 'embedding_llama', 'embedding_collab']},
        {'targets': [{'representation': 'uid'}]},
    )
    assert list(graph['representations']) == [
        'embedding_collab', 'embedding_llama', 'sid_hybrid', 'uid',
    ]
    assert graph['representations']['sid_hybrid']['codec']['quantizer']['codebook_size'] == 128
    assert 'unused_sid' not in graph['representations']


def test_decoder_target_must_lead_encoder_order():
    with pytest.raises(ValueError, match='must lead'):
        normalize_representation_graph(
            _catalog(),
            {'representations': ['embedding_llama', 'uid']},
            {'targets': [{'representation': 'uid'}]},
        )


def test_train_config_v4_bridges_named_representations():
    root = {
        'schema_version': 'trainer.v4',
        'data': {'name': 'mindf'},
        'representations': _catalog(),
        'encoder': {
            'representations': ['uid', 'sid_hybrid', 'embedding_llama', 'embedding_collab'],
            'combine': 'concat',
            'max_items': 256,
        },
        'decoder': {'targets': [{'representation': 'uid', 'decoding': {'mode': 'flat'}}]},
        'model': {
            'name': 'scratch', 'dtype': 'auto', 'freeze_backbone': 'auto', 'max_length': 0,
            'lora': {'use': 'auto', 'rank': 8, 'alpha': 32, 'dropout': 0.05, 'layers': None},
            'scratch': {'hidden_size': 256, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.1},
        },
        'trainer': {
            'batch_size': 64, 'accumulate_batch': 1, 'epochs': 0, 'learning_rate': 0.0001,
            'weight_decay': 0.01, 'seed': 42, 'device': None,
        },
        'evaluator': {'main_metric': 'ndcg@10', 'patience': 3, 'metrics': ['ndcg@10']},
    }
    config = TrainConfig.from_refconfig(Obj({'config': root}))
    assert config.repr_type == 'uid+sid+embedding'
    assert config.task_type == 'uid'
    assert config.compile_config.representation_names == [
        'uid', 'sid_hybrid', 'embedding_llama', 'embedding_collab',
    ]


def test_model_renders_independent_markers_for_embedding_instances():
    graph = normalize_representation_graph(
        _catalog(),
        {'representations': ['uid', 'embedding_llama', 'embedding_collab']},
        {'targets': [{'representation': 'uid'}]},
    )

    class CompileConfig:
        representation_names = graph['encoder']['representations']

        @staticmethod
        def representation_kind(name):
            return graph['representations'][name]['type']

    class Config:
        repr_combine = 'concat'
        compile_config = CompileConfig()

    class Compiled:
        item_views = {
            'embedding_llama': [3],
            'embedding_collab': [7],
        }

    model = SequentialRecModel.__new__(SequentialRecModel)
    model.config = Config()
    model.compiled = Compiled()
    assert model._render_history_item(0) == [
        ('type_marker', 'uid'),
        ('uid', 0),
        ('type_marker', 'embedding_llama'),
        ('embedding', ('embedding_llama', 3)),
        ('type_marker', 'embedding_collab'),
        ('embedding', ('embedding_collab', 7)),
    ]


def test_model_initializes_independent_embedding_tables_and_projections():
    config = _load_profile(
        'embedding-dual.yaml',
        hidden_size=16,
        num_layers=1,
        num_heads=2,
    )
    compiled = SimpleNamespace(
        model_kind='scratch',
        model_vocab_size=8,
        meta={'model_max_length': 64},
        special_vocab={
            'tokens': ['<repr:uid>', '<repr:embedding_content>', '<repr:embedding_collaborative>'],
            'marker_to_index': {'uid': 0, 'embedding_content': 1, 'embedding_collaborative': 2},
        },
        num_items=3,
        sid_vocab_size=0,
        sid_num_quantizers=0,
        hash_vocab_size=0,
        hash_num_tokens=0,
        embedding_matrices={
            'embedding_content': torch.randn(3, 5),
            'embedding_collaborative': torch.randn(3, 2),
        },
        item_views={
            'uid': [0, 1, 2],
            'embedding_content': [0, 1, 2],
            'embedding_collaborative': [0, 1, 2],
        },
        finetune=pd.DataFrame({'sequence_uids': [[0, 1, 2]]}),
    )

    model = SequentialRecModel(compiled, config)

    assert set(model.embedding_tables) == {'embedding_content', 'embedding_collaborative'}
    assert model.embedding_projections['embedding_content'].in_features == 5
    assert model.embedding_projections['embedding_collaborative'].in_features == 2
    assert model._embed_spec('embedding', ('embedding_content', 1)).shape == (1, 16)


def _load_profile(name, **kwargs):
    return TrainConfig.from_refconfig(ConfigInit([], {}, []).parse_kwargs({
        'config': str(Path('config/trainer') / name),
        'data': 'mindf',
        'model': 'scratch',
        **kwargs,
    }))


def test_profiles_use_refconfig_native_multilevel_imports():
    profile = yaml.safe_load(Path('config/trainer/multi-decoder.yaml').read_text())
    assert profile['$$import'] == 'hybrid.yaml'
    assert 'extends' not in profile

    config = _load_profile('multi-decoder.yaml')
    assert config.batch_size == 64
    assert list(config.representation_graph['representations']) == [
        'embedding_collaborative',
        'embedding_content',
        'sid_hybrid',
        'uid',
    ]
    assert [target['representation'] for target in config.representation_graph['decoder']['targets']] == [
        'sid_hybrid',
        'uid',
    ]


def test_profiles_prune_catalog_and_inactive_parameters_do_not_change_sign():
    baseline = _load_profile('uid.yaml')
    changed_inactive = _load_profile('uid.yaml', content_embedding_model='qwen3embedding06b')

    assert list(baseline.representation_graph['representations']) == ['uid']
    assert trained_signature_from_config(baseline) == trained_signature_from_config(changed_inactive)


def test_active_named_source_changes_sign():
    baseline = _load_profile('sid-content.yaml', content_embedding_model='llama3')
    changed = _load_profile('sid-content.yaml', content_embedding_model='qwen3embedding06b')

    assert trained_signature_from_config(baseline) != trained_signature_from_config(changed)


def test_semantic_contract_ignores_instance_names_but_not_sources():
    old = normalize_representation_graph(
        {'embedding_old': {'type': 'embedding', 'sources': [{'model': 'llama3'}]}},
        {'representations': ['embedding_old']},
        {'targets': [{'representation': 'embedding_old'}]},
    )
    renamed = normalize_representation_graph(
        {'embedding_new': {'type': 'embedding', 'sources': [{'model': 'llama3'}]}},
        {'representations': ['embedding_new']},
        {'targets': [{'representation': 'embedding_new'}]},
    )
    changed = normalize_representation_graph(
        {'embedding_new': {'type': 'embedding', 'sources': [{'model': 'word2vec'}]}},
        {'representations': ['embedding_new']},
        {'targets': [{'representation': 'embedding_new'}]},
    )

    assert semantic_graph_contract(old) == semantic_graph_contract(renamed)
    assert semantic_graph_contract(old) != semantic_graph_contract(changed)


def test_legacy_uid_and_v4_uid_have_same_identity_when_values_match():
    legacy = TrainConfig.from_refconfig(ConfigInit([], {}, []).parse_kwargs({
        'config': 'config/trainer.yaml',
        'data': 'mindf',
        'model': 'scratch',
        'task_type': 'uid',
        'repr_type': 'uid',
        'maxitems': 50,
    }))
    current = _load_profile('uid.yaml', maxitems=50)

    assert semantic_graph_contract(legacy.representation_graph) == semantic_graph_contract(current.representation_graph)
    assert legacy.sign_payload == current.sign_payload
    assert trained_signature_from_config(legacy) == trained_signature_from_config(current)

    trainer = Trainer.__new__(Trainer)
    trainer.config = current
    trainer._assert_checkpoint_compatible({'config': asdict(legacy), 'model_state_dict': {}})


def test_legacy_add_fusion_keeps_dedicated_runtime_path():
    config = TrainConfig.from_refconfig(ConfigInit([], {}, []).parse_kwargs({
        'config': 'config/trainer.yaml',
        'data': 'mindf',
        'model': 'scratch',
        'task_type': 'uid',
        'repr_type': 'uid+embedding',
        'repr_source_model': 'llama3',
        'repr_combine': 'add',
    }))

    assert config.representation_graph is None
    assert config.compile_config.representation_names == ['uid', 'embedding']


def test_checkpoint_adapter_maps_renamed_markers_and_embedding_modules():
    current = _load_profile('embedding-dual.yaml')
    saved_config = asdict(current)
    saved_graph = deepcopy(saved_config['representation_graph'])
    renames = {
        'uid': 'old_uid',
        'embedding_content': 'old_content',
        'embedding_collaborative': 'old_collaborative',
    }
    saved_graph['representations'] = {
        renames[name]: spec for name, spec in saved_graph['representations'].items()
    }
    saved_graph['encoder']['representations'] = [
        renames[name] for name in saved_graph['encoder']['representations']
    ]
    for target in saved_graph['decoder']['targets']:
        target['representation'] = renames[target['representation']]
    saved_config['representation_graph'] = saved_graph

    current_state = {
        'type_marker_embedding.weight': torch.zeros((3, 2)),
        'embedding_projections.embedding_content.weight': torch.zeros((2, 4)),
        'embedding_projections.embedding_collaborative.weight': torch.zeros((2, 3)),
        'embedding_heads.embedding_content.weight': torch.zeros((4, 2)),
        'embedding_heads.embedding_collaborative.weight': torch.zeros((3, 2)),
    }

    class Model:
        @staticmethod
        def state_dict():
            return current_state

    trainer = Trainer.__new__(Trainer)
    trainer.config = current
    trainer.model_core = Model()
    trainer.compiled = SimpleNamespace(special_vocab={
        'marker_to_index': {
            'uid': 0,
            'embedding_content': 1,
            'embedding_collaborative': 2,
        },
    })
    old_markers = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    adapted = trainer._adapt_checkpoint_state_dict({
        'type_marker_embedding.weight': old_markers,
        'embedding_projections.old_content.weight': torch.ones((2, 4)),
        'embedding_projections.old_collaborative.weight': torch.ones((2, 3)) * 2,
        'embedding_heads.old_content.weight': torch.ones((4, 2)) * 3,
        'embedding_heads.old_collaborative.weight': torch.ones((3, 2)) * 4,
    }, saved_config)

    assert torch.equal(adapted['type_marker_embedding.weight'], old_markers)
    assert torch.equal(adapted['embedding_projections.embedding_content.weight'], torch.ones((2, 4)))
    assert torch.equal(adapted['embedding_projections.embedding_collaborative.weight'], torch.ones((2, 3)) * 2)
    assert torch.equal(adapted['embedding_heads.embedding_content.weight'], torch.ones((4, 2)) * 3)
    assert torch.equal(adapted['embedding_heads.embedding_collaborative.weight'], torch.ones((3, 2)) * 4)


def test_checkpoint_rejects_changed_representation_source():
    saved = _load_profile('sid-content.yaml', content_embedding_model='llama3')
    current = _load_profile('sid-content.yaml', content_embedding_model='qwen3embedding06b')
    trainer = Trainer.__new__(Trainer)
    trainer.config = current

    with pytest.raises(ValueError, match='representation_graph'):
        trainer._assert_checkpoint_compatible({
            'config': asdict(saved),
            'model_state_dict': {},
        })
