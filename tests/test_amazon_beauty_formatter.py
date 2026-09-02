import gzip
import json

import pytest
import pandas as pd

from formatters.automotive_formatter import AutomotiveFormatter
from formatters.beauty_formatter import BeautyFormatter
from formatters.books_formatter import BooksFormatter
from formatters.cds_formatter import CDsFormatter


def _write_jsonl_gzip(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row) + '\n')


def test_beauty_formatter_reads_all_beauty_files(tmp_path):
    _write_jsonl_gzip(
        tmp_path / 'meta_All_Beauty.json.gz',
        [
            {'asin': 'A1', 'title': 'Face &amp; Wash!'},
            {'asin': 'A2', 'title': 'Night Cream'},
        ],
    )
    _write_jsonl_gzip(
        tmp_path / 'All_Beauty.json.gz',
        [
            {'reviewerID': 'U1', 'asin': 'A2', 'overall': 5, 'reviewTime': '02 01, 2020'},
            {'reviewerID': 'U1', 'asin': 'A1', 'overall': 4, 'reviewTime': '01 01, 2020'},
            {'reviewerID': 'U1', 'asin': 'A2', 'overall': 2, 'reviewTime': '03 01, 2020'},
        ],
    )

    formatter = BeautyFormatter(data_dir=tmp_path)
    formatter.items = formatter.load_items()
    users = formatter.load_users()

    assert formatter.items['asin'].tolist() == ['A1', 'A2']
    assert users.to_dict('records') == [{'reviewerID': 'U1', 'history': ['A1', 'A2']}]


@pytest.mark.parametrize(
    ('formatter_class', 'subset'),
    [
        (BeautyFormatter, 'All_Beauty'),
        (CDsFormatter, 'CDs_and_Vinyl'),
        (BooksFormatter, 'Books'),
        (AutomotiveFormatter, 'Automotive'),
    ],
)
def test_amazon_formatters_use_explicit_dataset_filenames(tmp_path, formatter_class, subset):
    formatter = formatter_class(data_dir=tmp_path)
    assert formatter.subset == subset
    formatter.items = pd.DataFrame({'asin': []})

    with pytest.raises(FileNotFoundError, match=f'{subset}.json.gz'):
        formatter.load_users()
