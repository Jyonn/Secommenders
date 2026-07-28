import pandas as pd

from formatters.mind_formatter import MINDFFormatter
from formatters.recif_ads_formatter import RAFFormatter
from utils.class_hub import ClassHub


def test_full_formatters_are_registered():
    formatters = ClassHub.formatters().class_dict

    assert formatters['mindf'] is MINDFFormatter
    assert formatters['raf'] is RAFFormatter


def test_mindf_keeps_latest_user_history_and_drops_unused_items(monkeypatch):
    items = pd.DataFrame(
        {
            'nid': [f'n{index}' for index in range(305)] + ['unused'],
            'cat': ['cat'] * 306,
            'subcat': ['subcat'] * 306,
            'title': ['title'] * 306,
            'abs': ['abstract'] * 306,
        }
    )
    users = pd.DataFrame(
        [
            {
                'uid': 'u1',
                'time': '2026-01-01',
                'history': 'n0 n1 n2',
            },
            {
                'uid': 'u1',
                'time': '2026-01-02',
                'history': ' '.join(f'n{index}' for index in range(300)),
            },
            {
                'uid': 'u2',
                'time': '2026-01-03',
                'history': 'missing',
            },
        ]
    )

    monkeypatch.setattr(MINDFFormatter, '_load_raw_items', lambda self: items)
    monkeypatch.setattr(MINDFFormatter, '_load_raw_user_rows', lambda self: users)

    formatter = MINDFFormatter(data_dir='unused')
    formatted_items = formatter.load_items()
    formatted_users = formatter.load_users()

    assert len(formatted_users) == 1
    assert formatted_users.iloc[0]['history'] == [f'n{index}' for index in range(44, 300)]
    assert set(formatted_items['nid']) == set(formatted_users.iloc[0]['history'])
    assert formatter._extra_stats()['item_count_before'] == 306
    assert formatter._extra_stats()['item_count_after'] == 256
    assert formatter._extra_stats()['user_count_before'] == 2
    assert formatter._extra_stats()['user_count_after'] == 1
    assert formatter._extra_stats()['interaction_count_after'] == 256
    assert formatter._extra_stats()['user_sequence_length_before']['summary']['count'] == 3
    assert formatter._extra_stats()['user_sequence_length_after']['summary']['max'] == 256


def test_raf_disables_ncore_and_keeps_latest_history(monkeypatch):
    long_history = list(range(300))
    raw_users = pd.DataFrame(
        {
            'uid': ['u1', 'u2'],
            'history': [long_history, [999]],
        }
    )

    monkeypatch.setattr(RAFFormatter, '_load_raw_users', lambda self: raw_users)
    monkeypatch.setattr(RAFFormatter, '_stream_caption_pid_set', lambda self: set(long_history + [999]))
    monkeypatch.setattr(
        RAFFormatter,
        '_load_caption_rows',
        lambda self, item_ids: pd.DataFrame(
            {
                'pid': sorted(item_ids),
                'caption': [f'caption-{item_id}' for item_id in sorted(item_ids)],
            }
        ),
    )

    formatter = RAFFormatter(data_dir='unused')
    formatted_items = formatter.load_items()
    formatted_users = formatter.load_users()

    assert formatter.max_length == 256
    assert len(formatted_users) == 2
    assert formatted_users.iloc[0]['history'] == list(range(44, 300))
    assert formatted_users.iloc[1]['history'] == [999]
    assert 999 in set(formatted_items['pid'])
    assert formatter._extra_stats()['item_count_before'] == 301
    assert formatter._extra_stats()['item_count_after'] == 257
    assert formatter._extra_stats()['interaction_count_after'] == 257
    assert formatter._extra_stats()['user_sequence_length_before']['summary']['max'] == 300
    assert formatter._extra_stats()['user_sequence_length_after']['summary']['max'] == 256
