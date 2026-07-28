from processors.base_processor import Processor


def test_parquet_magic_byte_check_rejects_non_parquet_file(tmp_path):
    path = tmp_path / 'finetune.parquet'
    path.write_text('not a parquet file')

    assert not Processor._parquet_file_looks_valid(path)


def test_parquet_magic_byte_check_accepts_basic_parquet_shape(tmp_path):
    path = tmp_path / 'finetune.parquet'
    path.write_bytes(b'PAR1payloadPAR1')

    assert Processor._parquet_file_looks_valid(path)
