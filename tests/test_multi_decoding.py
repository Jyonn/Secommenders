import unittest
from utils.compile import CompileConfig, canonicalize_task_type
from utils.artifact_identity import migrate_train_config_dict, trained_spec_from_config
from utils.multi_decoding import fuse_candidate_scores, uid_frequency_gate


class MultiDecodingTests(unittest.TestCase):
    def test_task_order_is_canonical(self):
        self.assertEqual(canonicalize_task_type('uid+sid'), 'sid+uid')
        self.assertEqual(canonicalize_task_type('sid+uid'), 'sid+uid')

        config = CompileConfig(
            data='mindf',
            model='scratch',
            repr_type='uid+sid',
            repr_source_model='llama3',
            sid_export='coll',
            sid_coder='rqvae',
            hash_coder=None,
            task_type='uid+sid',
            maxitems=50,
        )
        self.assertEqual(config.task_type, 'sid+uid')
        self.assertEqual(config.repr_type, 'sid+uid')
        self.assertEqual(config.task_types, ['sid', 'uid'])
        self.assertEqual(config.used_views, {'sid', 'uid'})

    def test_frequency_gate_prefers_sid_for_cold_and_uid_for_warm(self):
        kwargs = dict(mode='frequency', uid_weight=0.5, threshold=5, smoothing=0.5)
        self.assertLess(uid_frequency_gate(0, **kwargs), 0.5)
        self.assertGreater(uid_frequency_gate(100, **kwargs), 0.5)

    def test_fixed_fusion_is_frequency_independent(self):
        kwargs = dict(mode='fixed', uid_weight=0.5, threshold=5, smoothing=0.5)
        self.assertEqual(uid_frequency_gate(0, **kwargs), 0.5)
        self.assertEqual(uid_frequency_gate(100, **kwargs), 0.5)

    def test_frequency_fusion_can_select_sid_cold_and_uid_warm_items(self):
        ranked = fuse_candidate_scores(
            uid_scores={0: -5.0, 1: 5.0},
            sid_scores={0: 5.0, 1: -5.0},
            frequencies={0: 0, 1: 100},
            fusion_mode='frequency',
            uid_weight=0.5,
            score_normalization='zscore',
            temperature_uid=1.0,
            temperature_sid=1.0,
            frequency_threshold=5,
            frequency_smoothing=0.5,
            output_topk=2,
        )
        self.assertEqual({uid for uid, _ in ranked}, {0, 1})
        self.assertTrue(all(score > 0 for _, score in ranked))

    def test_registry_migration_omits_multi_defaults_for_single_task(self):
        migrated = migrate_train_config_dict({
            'data': 'mindf',
            'model': 'scratch',
            'repr_type': 'uid',
            'task_type': 'uid',
        })
        sign_config = trained_spec_from_config(migrated)['config']
        self.assertFalse(any(key.startswith('multi_') for key in sign_config))

    def test_registry_migration_canonicalizes_multi_task_order(self):
        common = {
            'data': 'mindf',
            'model': 'scratch',
            'repr_source_model': 'llama3',
            'sid_coder': 'rqvae',
            'sid_export': 'coll',
        }
        first = migrate_train_config_dict({**common, 'repr_type': 'uid+sid', 'task_type': 'uid+sid'})
        second = migrate_train_config_dict({**common, 'repr_type': 'sid+uid', 'task_type': 'sid+uid'})
        first_spec = trained_spec_from_config(first)
        second_spec = trained_spec_from_config(second)
        self.assertEqual(first_spec, second_spec)
        self.assertEqual(first_spec['config']['task_type'], 'sid+uid')
        self.assertIn('multi_fusion', first_spec['config'])


if __name__ == '__main__':
    unittest.main()
