from types import SimpleNamespace

import pandas as pd

from compiler import Compiler
from core.dataset import CompiledFinetuneTrajectoryDataset, CompiledTestSampleDataset


def _compiler(tmp_path):
    compiler = Compiler.__new__(Compiler)
    compiler.config = SimpleNamespace(task_type='uid', repr_type='uid')
    compiler.processor = SimpleNamespace(UID_COL='reviewerID', HIS_COL='history', multi_item_col=None)
    compiler.uid_item_map = {'A': 0, 'B': 1, 'C': 2}
    compiler.samples_dir = tmp_path
    compiler.samples_stats = {}
    compiler.sample_visuals = {}
    compiler._build_usable_sequence = lambda sequence: (sequence, len(sequence))
    compiler._build_usable_history = lambda history, target: (history, len(history) + 1)
    compiler._render_sample_visual = lambda **kwargs: 'preview'
    return compiler


def test_compiler_canonicalizes_finetune_user_column(tmp_path):
    compiler = _compiler(tmp_path)
    source = pd.DataFrame({'reviewerID': ['U1'], 'history': [['A', 'B']]})

    compiler.build_samples('finetune', source)

    compiled = pd.read_parquet(tmp_path / 'finetune.parquet')
    assert list(compiled.columns) == [
        'uid', 'sequence_uids', 'sequence_item_count', 'prediction_count', 'total_input_length',
    ]
    assert CompiledFinetuneTrajectoryDataset(compiled)[0]['uid'] == 'U1'


def test_compiler_canonicalizes_evaluation_user_column(tmp_path):
    compiler = _compiler(tmp_path)
    source = pd.DataFrame({'reviewerID': ['U1'], 'history': [['A', 'B', 'C']]})

    compiler.build_samples('test', source)

    compiled = pd.read_parquet(tmp_path / 'test.parquet')
    assert compiled.loc[0, 'uid'] == 'U1'
    assert CompiledTestSampleDataset(compiled)[0]['uid'] == 'U1'
