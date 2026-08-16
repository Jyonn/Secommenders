import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:
    pyarrow = None

from scripts.migrate_word2vec_embeddings import migrate
from utils.word2vec import word2vec_model_ref


class Word2VecMigrationTest(unittest.TestCase):
    @unittest.skipUnless(pyarrow is not None, 'pyarrow is required for parquet migration tests')
    def test_extracts_pure_collaborative_cluster_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'artifacts/clustered/mindf/old-sign'
            source.mkdir(parents=True)
            embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.save(source / 'item_embeddings.npy', embeddings)
            pd.DataFrame({'item_id': ['a', 'b', 'c']}).to_parquet(source / 'item_ids.parquet', index=False)
            meta = {
                'dataset': 'mindf',
                'embedding': {'source': 'collaborative'},
                'processed_items_path': 'artifacts/processed/mindf/items.parquet',
                'word2vec': {'vector_size': 4, 'window': 5, 'patience': 5},
            }
            (source / 'meta.json').write_text(json.dumps(meta))

            self.assertEqual(migrate(root, 'mindf', apply=True), 0)
            target = root / 'artifacts/embedded/mindf' / word2vec_model_ref(meta['word2vec'])
            np.testing.assert_array_equal(np.load(target / 'embeddings.npy'), embeddings)
            self.assertEqual(json.loads((target / 'meta.json').read_text())['source'], 'migrated-clustered-word2vec')

    def test_does_not_extract_fused_cluster_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'artifacts/clustered/mindf/mixed'
            source.mkdir(parents=True)
            (source / 'meta.json').write_text(json.dumps({
                'dataset': 'mindf',
                'embedding': {'source': 'concat'},
            }))
            self.assertEqual(migrate(root, 'mindf', apply=True), 0)
            self.assertFalse((root / 'artifacts/embedded').exists())


if __name__ == '__main__':
    unittest.main()
