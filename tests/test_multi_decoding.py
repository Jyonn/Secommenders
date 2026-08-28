import unittest
import torch
import torch.nn.functional as F

from core.model import SequentialRecModel
from utils.compile import CompileConfig, canonicalize_task_type
from utils.artifact_identity import migrate_train_config_dict, trained_spec_from_config
from utils.multi_decoding import fuse_candidate_scores


class MultiDecodingTests(unittest.TestCase):
    def test_sequential_sid_rescores_every_union_candidate_with_teacher_forcing(self):
        model = SequentialRecModel.__new__(SequentialRecModel)
        torch.nn.Module.__init__(model)
        model.dummy_parameter = torch.nn.Parameter(torch.zeros(1))
        item_codes = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        model._resolve_sid_name = lambda representation=None: 'sid_content'
        model._sid_item_codes = lambda representation=None: item_codes
        model._mask_sid_logits_for_slots = lambda logits, slots, representation=None: logits

        def predict(sample, prefixes, slot_index, representation=None):
            if slot_index == 0:
                return torch.tensor([[2.0, 1.0, 0.0, 0.0]])
            rows = []
            for prefix in prefixes:
                rows.append([0.0, 0.0, 3.0, 1.0] if prefix == [0] else [0.0, 0.0, 1.0, 3.0])
            return torch.tensor(rows)

        model._predict_sid_step_logits = predict
        scores = model._score_sequential_sid_candidates({}, [0, 1], 'sid_content')

        slot0 = F.log_softmax(torch.tensor([2.0, 1.0, 0.0, 0.0]), dim=-1)
        slot1_for_zero = F.log_softmax(torch.tensor([0.0, 0.0, 3.0, 1.0]), dim=-1)
        slot1_for_one = F.log_softmax(torch.tensor([0.0, 0.0, 1.0, 3.0]), dim=-1)
        self.assertAlmostEqual(scores[0], float(slot0[0] + slot1_for_zero[2]), places=6)
        self.assertAlmostEqual(scores[1], float(slot0[1] + slot1_for_one[3]), places=6)

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

    def test_fixed_fusion_combines_complete_candidate_scores(self):
        ranked = fuse_candidate_scores(
            uid_scores={0: -5.0, 1: 5.0},
            sid_scores={0: 5.0, 1: -5.0},
            uid_weight=0.5,
            score_normalization='zscore',
            temperature_uid=1.0,
            temperature_sid=1.0,
            output_topk=2,
        )
        self.assertEqual({uid for uid, _ in ranked}, {0, 1})
        self.assertTrue(all(score == 0 for _, score in ranked))

    def test_fusion_rejects_candidates_without_complete_scores(self):
        with self.assertRaisesRegex(ValueError, 'complete scores'):
            fuse_candidate_scores(
                uid_scores={0: 1.0, 1: 0.5},
                sid_scores={0: 1.0},
                uid_weight=0.5,
                score_normalization='none',
                temperature_uid=1.0,
                temperature_sid=1.0,
                output_topk=2,
            )

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
