import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.embedding_fusion import (
    embedding_fusion_from_flat,
    fusion_model_ref,
    is_legacy_single_source,
    normalize_embedding_fusion,
    load_fused_embeddings,
)
from utils.artifact_identity import (
    compiled_signature_from_config,
    quantized_spec_from_config,
    quantized_spec_from_upstream,
    trained_signature_from_config,
)
from utils.artifact import ArtifactStore
from utils.compile import CompileConfig


class EmbeddingFusionTest(unittest.TestCase):
    def test_flat_parameters_expand_into_sources(self):
        value = embedding_fusion_from_flat(
            'llama3,word2vec',
            normalize='true,true',
            reduce_dims='256,64',
            weights='1,0.5',
            word2vec_config={'vector_size': 32, 'window': 10},
        )
        spec = normalize_embedding_fusion(value)
        self.assertEqual([source['reduce_dim'] for source in spec['sources']], [256, 64])
        self.assertEqual([source['weight'] for source in spec['sources']], [1.0, 0.5])
        self.assertTrue(spec['sources'][1]['model'].startswith('word2vec/'))

    def test_flat_parameters_reject_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, 'embedding_reduce_dims'):
            embedding_fusion_from_flat('a,b,c', reduce_dims='64,32')

    def test_legacy_single_source_keeps_model_identity(self):
        spec = normalize_embedding_fusion({}, legacy_model='llama3')
        self.assertTrue(is_legacy_single_source(spec))
        self.assertEqual(fusion_model_ref(spec), 'llama3')

    def test_source_order_and_transform_change_identity(self):
        first = normalize_embedding_fusion({'sources': [
            {'model': 'llama3', 'reduce_dim': 128},
            {'model': 'word2vec'},
        ]})
        reversed_spec = normalize_embedding_fusion({'sources': list(reversed(first['sources']))})
        weighted = normalize_embedding_fusion({'sources': [
            {'model': 'llama3', 'reduce_dim': 128, 'weight': 2},
            {'model': 'word2vec'},
        ]})
        self.assertNotEqual(fusion_model_ref(first), fusion_model_ref(reversed_spec))
        self.assertNotEqual(fusion_model_ref(first), fusion_model_ref(weighted))

    def test_fusion_seed_does_not_change_identity(self):
        value = {'sources': [{'model': 'llama3'}, {'model': 'word2vec'}]}
        first = normalize_embedding_fusion({**value, 'seed': 7})
        second = normalize_embedding_fusion({**value, 'seed': 42})
        self.assertEqual(fusion_model_ref(first), fusion_model_ref(second))

    def test_word2vec_config_resolves_signed_source(self):
        spec = normalize_embedding_fusion({'sources': [{
            'model': 'word2vec',
            'config': {'vector_size': 32, 'window': 10},
        }]})
        self.assertTrue(spec['sources'][0]['model'].startswith('word2vec/'))

    def test_legacy_quantized_payload_is_unchanged(self):
        base = {
            'embedding_model': 'llama3',
            'quantizer': {'name': 'rqvae', 'config': {}},
            'encoder': {'name': 'mlp', 'config': {}},
            'trainer': {},
        }
        explicit = {
            **base,
            'embedding': normalize_embedding_fusion({}, legacy_model='llama3'),
        }
        self.assertEqual(
            quantized_spec_from_upstream('mindf', base),
            quantized_spec_from_upstream('mindf', explicit),
        )

    def test_load_aligns_normalizes_weights_and_concatenates(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ArtifactStore, 'ROOT', Path(directory)):
            first_dir = ArtifactStore('mindf').embedded_dir('first')
            second_dir = ArtifactStore('mindf').embedded_dir('second')
            np.save(first_dir / 'embeddings.npy', np.asarray([[3, 4], [0, 2]], dtype=np.float32))
            np.save(second_dir / 'embeddings.npy', np.asarray([[0, 5], [12, 0]], dtype=np.float32))
            frames = {
                str(first_dir / 'item_ids.parquet'): pd.DataFrame({'item_id': ['a', 'b']}),
                str(second_dir / 'item_ids.parquet'): pd.DataFrame({'item_id': ['b', 'a']}),
            }

            class Processor:
                IID_COL = 'item_id'
                items = pd.DataFrame({'item_id': ['a', 'b']})

            spec = normalize_embedding_fusion({'sources': [
                {'model': 'first', 'weight': 1},
                {'model': 'second', 'weight': 1},
            ]})
            with patch('pandas.read_parquet', side_effect=lambda path: frames[str(path)]):
                fused, item_ids, _ = load_fused_embeddings('mindf', Processor(), spec, lambda *args, **kwargs: None)
            scale = np.sqrt(0.5)
            expected = np.asarray([
                [0.6, 0.8, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ], dtype=np.float32) * scale
            self.assertEqual(item_ids, ['a', 'b'])
            np.testing.assert_allclose(fused, expected, atol=1e-6)

    def test_upstream_and_runtime_config_produce_same_quantized_spec(self):
        embedding = normalize_embedding_fusion({'sources': [
            {'model': 'llama3', 'reduce_dim': 128, 'weight': 1},
            {'model': 'word2vec', 'weight': 1, 'config': {'vector_size': 32, 'window': 10}},
        ]})
        upstream = {
            'embedding_model': None,
            'embedding': embedding,
            'quantizer': {'name': 'rqvae', 'config': {}},
            'encoder': {'name': 'mlp', 'config': {}},
            'trainer': {},
        }
        runtime = {
            'embedding': {
                'models': ','.join(source['model'] for source in embedding['sources']),
                'normalize': ','.join(str(source['normalize']).lower() for source in embedding['sources']),
                'reduce_dims': ','.join(str(source['reduce_dim']) for source in embedding['sources']),
                'weights': ','.join(str(source['weight']) for source in embedding['sources']),
                'fusion': embedding['fusion'],
                'word2vec': embedding['sources'][1]['config'],
            },
            'quantizer': upstream['quantizer'],
            'encoder': upstream['encoder'],
            'trainer': upstream['trainer'],
        }
        self.assertEqual(
            quantized_spec_from_upstream('mindf', upstream),
            quantized_spec_from_config('mindf', fusion_model_ref(embedding), runtime),
        )

    def test_direct_embedding_fusion_changes_compiled_and_trained_signatures(self):
        base = dict(
            data='mindf',
            model='scratch',
            repr_type='uid+embedding',
            repr_source_model=None,
            sid_export=None,
            sid_coder=None,
            hash_coder=None,
            task_type='uid',
            maxitems=50,
            repr_combine='concat',
        )
        first_embedding = normalize_embedding_fusion({'sources': [
            {'model': 'llama3', 'reduce_dim': 128},
            {'model': 'word2vec', 'reduce_dim': 64},
        ]})
        second_embedding = normalize_embedding_fusion({'sources': [
            {'model': 'llama3', 'reduce_dim': 256},
            {'model': 'word2vec', 'reduce_dim': 64},
        ]})
        first = CompileConfig(**base, embedding=first_embedding)
        second = CompileConfig(**base, embedding=second_embedding)
        self.assertNotEqual(compiled_signature_from_config(first), compiled_signature_from_config(second))

        train_base = {
            **base,
            'effective_batch_size': 64,
            'upstreams': {},
        }
        self.assertNotEqual(
            trained_signature_from_config({**train_base, 'repr_embedding': first_embedding}),
            trained_signature_from_config({**train_base, 'repr_embedding': second_embedding}),
        )


if __name__ == '__main__':
    unittest.main()
