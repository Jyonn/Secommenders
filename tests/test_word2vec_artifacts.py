import unittest

from utils.word2vec import normalize_word2vec_config, word2vec_model_ref, word2vec_signature


class Word2VecArtifactTest(unittest.TestCase):
    def test_default_config_has_stable_nested_artifact_ref(self):
        config = normalize_word2vec_config()
        self.assertEqual(word2vec_model_ref(config), f'word2vec/{word2vec_signature(config)}')
        self.assertEqual(len(word2vec_signature(config)), 16)

    def test_seed_and_workers_do_not_change_signature(self):
        baseline = word2vec_signature()
        self.assertEqual(baseline, word2vec_signature({'seed': 7, 'workers': 32}))

    def test_training_semantics_change_signature(self):
        baseline = word2vec_signature()
        self.assertNotEqual(baseline, word2vec_signature({'window': 10}))
        self.assertNotEqual(baseline, word2vec_signature({'learning_rate': 0.001}))
        self.assertNotEqual(baseline, word2vec_signature({'max_epochs': 50}))


if __name__ == '__main__':
    unittest.main()
