from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import torch
import yaml
from oba import Obj

import core.compiled as compiled_module
from core.compiled import CompiledArtifacts
from core.train_config import TrainConfig
from core.model import SequentialRecModel
from compiler import Compiler, VocabularyRegistry
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


def test_content_sid_profile_keeps_dense_content_and_collaborative_inputs_separate():
    config = _load_profile('sid-content-dual-embedding.yaml')

    assert config.compile_config.representation_names == [
        'sid_content',
        'embedding_content',
        'embedding_collaborative',
    ]
    assert config.compile_config.target_names == ['sid_content']
    assert config.compile_config.names_for_kind('embedding') == [
        'embedding_content',
        'embedding_collaborative',
    ]
    sid_source = config.compile_config.upstream_for('sid_content')['embedding']['sources'][0]
    catalog = config.representation_graph['representations']
    assert sid_source['model'] == 'llama3'
    assert sid_source['reduce_dim'] == 0
    assert catalog['embedding_content']['embedding']['sources'][0]['reduce_dim'] == 256
    assert catalog['embedding_collaborative']['embedding']['sources'][0]['reduce_dim'] == 64

    hybrid = _load_profile('sid-hybrid.yaml')
    hybrid_sources = hybrid.compile_config.upstream_for('sid_hybrid')['embedding']['sources']
    assert [source['reduce_dim'] for source in hybrid_sources] == [0, 0]


def test_multi_sid_profile_keeps_independent_upstreams_and_targets():
    config = _load_profile('sid-multi.yaml')

    assert config.is_multi_task
    assert config.compile_config.representation_names == ['sid_content', 'sid_collaborative']
    assert config.compile_config.target_names == ['sid_content', 'sid_collaborative']
    assert set(config.upstreams) == {'sid_content', 'sid_collaborative'}
    assert config.upstreams['sid_content']['embedding']['sources'][0]['model'] == 'llama3'
    assert config.upstreams['sid_collaborative']['embedding']['sources'][0]['model'].startswith('word2vec/')
    assert set(config.compile_config.compile_upstreams) == {'sid_content', 'sid_collaborative'}

    resized = _load_profile(
        'sid-multi.yaml',
        content_sid_codebook_size=256,
        collaborative_sid_codebook_size=64,
    )
    assert resized.upstreams['sid_content']['quantizer']['config']['codebook_size'] == 256
    assert resized.upstreams['sid_collaborative']['quantizer']['config']['codebook_size'] == 64
    assert trained_signature_from_config(config) != trained_signature_from_config(resized)

    hybrid = _load_profile('hybrid-multi-sid.yaml')
    assert hybrid.compile_config.representation_names == [
        'uid',
        'sid_content',
        'sid_collaborative',
        'embedding_content',
        'embedding_collaborative',
    ]
    assert hybrid.compile_config.target_names == ['uid']
    assert set(hybrid.upstreams) == {'sid_content', 'sid_collaborative'}


def test_model_initializes_independent_sid_embeddings_and_heads():
    config = _load_profile(
        'sid-multi.yaml',
        hidden_size=16,
        num_layers=1,
        num_heads=2,
    )
    sid_metadata = {
        'sid_content': {
            'vocab_size': 10,
            'num_quantizers': 2,
            'base_num_quantizers': 1,
            'codebook_size': 8,
            'collision_vocab_size': 2,
            'collision_token_offset': 8,
            'recommended_decoding': 'parallel',
        },
        'sid_collaborative': {
            'vocab_size': 7,
            'num_quantizers': 2,
            'base_num_quantizers': 1,
            'codebook_size': 5,
            'collision_vocab_size': 2,
            'collision_token_offset': 5,
            'recommended_decoding': 'parallel',
        },
    }
    compiled = SimpleNamespace(
        model_kind='scratch',
        model_vocab_size=8,
        meta={'model_max_length': 64},
        special_vocab={
            'tokens': [
                '<repr:sid_content>',
                '<repr:sid_collaborative>',
                '<repr:decoder_sid_content_sid_collaborative>',
            ],
            'marker_to_index': {
                'sid_content': 0,
                'sid_collaborative': 1,
                'decoder_sid_content_sid_collaborative': 2,
            },
        },
        prompt_main={
            'history_prefix_ids': [],
            'item_separator_ids': [],
            'query_prefix_ids': [],
        },
        num_items=2,
        hash_vocab_size=0,
        hash_num_tokens=0,
        embedding_matrices={},
        item_views={
            'sid_content': [[0, 8], [1, 8]],
            'sid_collaborative': [[0, 5], [1, 5]],
        },
        finetune=pd.DataFrame({'sequence_uids': [[0, 1]]}),
        sid_vocab_size_for=lambda name: sid_metadata[name]['vocab_size'],
        sid_metadata_for=lambda name: sid_metadata[name],
    )

    model = SequentialRecModel(compiled, config)

    assert set(model.sid_embeddings) == {'sid_content', 'sid_collaborative'}
    assert set(model.sid_heads) == {'sid_content', 'sid_collaborative'}
    assert model.sid_heads['sid_content'].out_features == 10
    assert model.sid_heads['sid_collaborative'].out_features == 7
    assert model._render_history_item(0) == [
        ('type_marker', 'sid_content'),
        ('sid', ('sid_content', [0, 8])),
        ('type_marker', 'sid_collaborative'),
        ('sid', ('sid_collaborative', [0, 5])),
    ]
    loss, metrics = model.forward_finetune_batch([{'sequence_uids': [0, 1]}])
    assert torch.isfinite(loss)
    assert 'sid_content_loss' in metrics
    assert 'sid_collaborative_loss' in metrics
    assert 'sid_token_acc' in metrics
    eval_loss, eval_metrics = model.forward_next_item_batch([{
        'history_uids': [0],
        'target_uid': 1,
        'ground_truth_uids': [1],
    }])
    assert torch.isfinite(eval_loss)
    assert 'sid_content_ndcg@10' in eval_metrics
    assert 'sid_collaborative_ndcg@10' in eval_metrics
    assert eval_metrics['ndcg@10'] == pytest.approx(
        (
            eval_metrics['sid_content_ndcg@10']
            + eval_metrics['sid_collaborative_ndcg@10']
        ) / 2
    )


def test_compiler_writes_one_vocabulary_per_sid_representation(tmp_path):
    config = _load_profile('sid-multi.yaml').compile_config
    compiler = Compiler.__new__(Compiler)
    compiler.config = config
    compiler.vocab_dir = tmp_path / 'vocab'
    compiler.prompts_dir = tmp_path / 'prompts'
    compiler.vocab_dir.mkdir()
    compiler.prompts_dir.mkdir()
    compiler.registry = VocabularyRegistry()
    compiler.uid_raw_items = ['a', 'b']
    compiler.backbone = SimpleNamespace(
        namespace_name='model',
        kind='scratch',
        build_vocab_artifact=lambda: {'tokens': ['x']},
        build_prompt_spec=lambda: {
            'history_prefix_ids': [],
            'item_separator_ids': [],
            'query_prefix_ids': [],
        },
    )
    compiler.load_sid_view = lambda name, build_only_meta=False: {
        'num_quantizers': 1,
        'final_num_quantizers': 2,
        'codebook_size': 8 if name == 'sid_content' else 5,
        'collision_vocab_size': 2,
        'collision_token_offset': 8 if name == 'sid_content' else 5,
        'collision_group_count': 0,
        'collided_item_count': 0,
        'max_collision_size': 2,
        'quantizer_name': 'rqvae',
        'quantizer_scheme': 'residual',
        'recommended_decoding': 'sequential',
        'quantized_export_dir': f'/tmp/{name}',
    }

    compiler.build_vocab_and_prompts()

    assert (compiler.vocab_dir / 'sid_content.json').exists()
    assert (compiler.vocab_dir / 'sid_collaborative.json').exists()
    sid_entries = [entry for entry in compiler.registry.entries if entry['kind'] == 'sid']
    assert [entry['name'] for entry in sid_entries] == ['sid_content', 'sid_collaborative']
    assert [entry['size'] for entry in sid_entries] == [10, 7]


def test_compiled_loader_builds_independent_sid_indices(tmp_path, monkeypatch):
    config = _load_profile('sid-multi.yaml')
    compile_dir = tmp_path / 'compiled'
    for name in ('samples', 'vocab', 'prompts', 'item_views'):
        (compile_dir / name).mkdir(parents=True)

    def write_json(relative, payload):
        (compile_dir / relative).write_text(json.dumps(payload))

    write_json('meta.json', {'model_kind': 'scratch', 'model_max_length': 64})
    write_json('vocab/uid.json', {'raw_item_ids': ['a', 'b']})
    write_json('vocab/special.json', {
        'tokens': ['<repr:sid_content>', '<repr:sid_collaborative>'],
        'marker_to_index': {'sid_content': 0, 'sid_collaborative': 1},
    })
    write_json('vocab/meta.json', {'namespaces': [
        {'name': 'model', 'kind': 'model', 'size': 1},
        {'name': 'sid_content', 'kind': 'sid', 'size': 10},
        {'name': 'sid_collaborative', 'kind': 'sid', 'size': 7},
    ]})
    write_json('vocab/sid_content.json', {
        'num_quantizers': 2,
        'base_num_quantizers': 1,
        'codebook_size': 8,
        'collision_vocab_size': 2,
        'collision_token_offset': 8,
        'recommended_decoding': 'parallel',
    })
    write_json('vocab/sid_collaborative.json', {
        'num_quantizers': 2,
        'base_num_quantizers': 1,
        'codebook_size': 5,
        'collision_vocab_size': 2,
        'collision_token_offset': 5,
        'recommended_decoding': 'parallel',
    })
    write_json('prompts/main.json', {
        'history_prefix_ids': [], 'item_separator_ids': [], 'query_prefix_ids': [],
    })
    write_json('item_views/meta.json', {
        'views': ['sid_content', 'sid_collaborative'],
        'types': {'sid_content': 'sid', 'sid_collaborative': 'sid'},
    })
    pd.DataFrame({'value': [[0, 8], [1, 8]]}).to_parquet(
        compile_dir / 'item_views/sid_content.parquet', index=False
    )
    pd.DataFrame({'value': [[0, 5], [1, 5]]}).to_parquet(
        compile_dir / 'item_views/sid_collaborative.parquet', index=False
    )
    sample = pd.DataFrame({'sequence_uids': [[0, 1]]})
    for split in ('finetune', 'valid', 'test'):
        sample.to_parquet(compile_dir / f'samples/{split}.parquet', index=False)

    monkeypatch.setattr(compiled_module, 'resolve_compiled_dir', lambda _config: compile_dir)
    compiled = CompiledArtifacts(config).load()

    assert set(compiled.sid_metadata) == {'sid_content', 'sid_collaborative'}
    assert compiled.sid_vocab_size_for('sid_content') == 10
    assert compiled.sid_vocab_size_for('sid_collaborative') == 7
    assert compiled.sid_prefix_to_next_by_name['sid_content'][()] == [0, 1]
    assert compiled.sid_prefix_to_next_by_name['sid_collaborative'][(0,)] == [5]
    assert compiled.item_views['sid'] is compiled.item_views['sid_content']


def test_checkpoint_adapter_maps_legacy_single_sid_modules_to_named_modules():
    config = _load_profile('sid-content.yaml')
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.compiled = SimpleNamespace(
        special_vocab={'marker_to_index': {'sid_content': 0}},
    )
    trainer.model_core = SimpleNamespace(state_dict=lambda: {
        'sid_embeddings.sid_content.weight': torch.zeros(10, 4),
        'sid_heads.sid_content.weight': torch.zeros(10, 4),
        'sid_heads.sid_content.bias': torch.zeros(10),
    })
    trainer._pnt = lambda *_args, **_kwargs: None
    state = {
        'sid_embedding.weight': torch.ones(10, 4),
        'sid_head.weight': torch.ones(10, 4),
        'sid_head.bias': torch.ones(10),
    }

    adapted = trainer._adapt_checkpoint_state_dict(state, asdict(config))

    assert 'sid_embedding.weight' not in adapted
    assert 'sid_head.weight' not in adapted
    assert torch.equal(adapted['sid_embeddings.sid_content.weight'], torch.ones(10, 4))
    assert torch.equal(adapted['sid_heads.sid_content.weight'], torch.ones(10, 4))
    assert torch.equal(adapted['sid_heads.sid_content.bias'], torch.ones(10))


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
