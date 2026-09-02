import gzip
import json

from formatters.beauty_formatter import BeautyFormatter


def _write_jsonl_gzip(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row) + '\n')


def test_beauty_formatter_reads_legacy_five_core_files(tmp_path):
    _write_jsonl_gzip(
        tmp_path / 'meta_Beauty.json.gz',
        [
            {'asin': 'A1', 'title': 'Face &amp; Wash!'},
            {'asin': 'A2', 'title': 'Night Cream'},
        ],
    )
    _write_jsonl_gzip(
        tmp_path / 'reviews_Beauty_5.json.gz',
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


def test_beauty_formatter_reads_amazon_2023_fields(tmp_path):
    _write_jsonl_gzip(
        tmp_path / 'raw_meta_Beauty.jsonl.gz',
        [
            {'parent_asin': 'P1', 'title': 'Cleanser'},
            {'parent_asin': 'P2', 'title': 'Serum'},
        ],
    )
    _write_jsonl_gzip(
        tmp_path / 'raw_review_Beauty.jsonl.gz',
        [
            {'user_id': 'U1', 'parent_asin': 'P2', 'rating': 5, 'timestamp': 2000},
            {'user_id': 'U1', 'parent_asin': 'P1', 'rating': 4, 'timestamp': 1000},
        ],
    )

    formatter = BeautyFormatter(data_dir=tmp_path)
    formatter.items = formatter.load_items()
    users = formatter.load_users()

    assert formatter.items['asin'].tolist() == ['P1', 'P2']
    assert users.to_dict('records') == [{'reviewerID': 'U1', 'history': ['P1', 'P2']}]


def test_beauty_formatter_reads_all_beauty_files(tmp_path):
    _write_jsonl_gzip(
        tmp_path / 'meta_All_Beauty.json.gz',
        [{'asin': 'A1', 'title': 'Face Wash'}],
    )
    _write_jsonl_gzip(
        tmp_path / 'All_Beauty.json.gz',
        [
            {'reviewerID': 'U1', 'asin': 'A1', 'overall': 5, 'unixReviewTime': 1000},
            {'reviewerID': 'U1', 'asin': 'A1', 'overall': 4, 'unixReviewTime': 2000},
        ],
    )

    formatter = BeautyFormatter(data_dir=tmp_path)
    formatter.items = formatter.load_items()
    users = formatter.load_users()

    assert formatter.items['asin'].tolist() == ['A1']
    assert users.to_dict('records') == [{'reviewerID': 'U1', 'history': ['A1', 'A1']}]
