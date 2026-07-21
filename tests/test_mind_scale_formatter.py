import pandas as pd

from formatters.mind_scale_formatter import MINDS1Formatter, MINDS10Formatter, MINDS99Formatter


def _formatter(formatter_class):
    formatter = formatter_class.__new__(formatter_class)
    formatter._scaled_test_users = None
    formatter._scale_total_users = None
    formatter._scale_test_start = None
    formatter._scale_train_limit = None
    return formatter


def test_scale_split_points_share_test_boundary_and_nest_training_prefixes():
    total = 10_000
    formatters = [_formatter(cls) for cls in (MINDS1Formatter, MINDS10Formatter, MINDS99Formatter)]
    points = [formatter._scale_split_points(total) for formatter in formatters]

    assert [point[0] for point in points] == [9_970, 9_970, 9_970]
    assert [point[1] for point in points] == [100, 1_000, 9_900]


def test_deduplicate_before_scaling_keeps_users_disjoint(monkeypatch):
    formatter = _formatter(MINDS10Formatter)
    raw_users = pd.DataFrame(
        [
            {'uid': f'u{index}', 'history': [f'n{index}'], 'time': pd.Timestamp('2026-01-01')}
            for index in range(1_000)
        ]
        + [
            {'uid': 'u0', 'history': ['newer'], 'time': pd.Timestamp('2026-01-02')},
        ]
    )

    monkeypatch.setattr(
        'formatters.mind_formatter.MINDFormatter.load_users',
        lambda self: raw_users,
    )

    train = formatter.load_users()
    test = formatter.load_test_users()

    assert len(train) == 100
    assert len(test) == 3
    assert set(train['uid']).isdisjoint(set(test['uid']))
    assert formatter._scale_total_users == 1_000

