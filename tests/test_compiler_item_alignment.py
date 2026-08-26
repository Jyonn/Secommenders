import numpy as np

from compiler import Compiler


def test_sid_view_aligns_integer_processor_ids_with_string_export_ids(tmp_path):
    compiler = Compiler.__new__(Compiler)
    compiler.uid_raw_items = [76, 125]
    compiler.sid_stats = {}
    compiler._load_quantized_export = lambda name: (
        tmp_path,
        {
            'quantizer_config': {'codebook_size': 128},
            'quantizer_model': 'rqvae',
        },
        ['76', '125'],
        np.asarray([[1, 2], [3, 4]], dtype=np.int64),
    )

    values = compiler.load_sid_view(name='sid_content')

    assert values == [[1, 130, 256], [3, 132, 256]]


def test_hash_view_aligns_integer_processor_ids_with_string_export_ids(tmp_path):
    compiler = Compiler.__new__(Compiler)
    compiler.uid_raw_items = [76, 125]
    compiler.hash_stats = {}
    compiler._load_hash_export = lambda: (
        tmp_path,
        {'quantizer_model': 'simhash'},
        ['76', '125'],
        np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.uint8),
    )

    values = compiler.load_hash_view()

    assert len(values) == 2
    assert values[0] != values[1]
