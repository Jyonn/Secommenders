import pandas as pd

from formatters.recif_scale_formatter import RecIFVideoSmallScaleFormatter
from formatters.recif_scale_formatter import RVT1Formatter, RVT10Formatter, RVT99Formatter
from utils.class_hub import ClassHub


def _formatter(formatter_class):
    return formatter_class.__new__(formatter_class)


def test_rvt_formatters_are_registered():
    formatters = ClassHub.formatters().class_dict

    assert formatters['rvt1'] is RVT1Formatter
    assert formatters['rvt10'] is RVT10Formatter
    assert formatters['rvt99'] is RVT99Formatter


def test_rvt_uses_tiny_filter_defaults_and_own_prefix():
    formatter = _formatter(RVT10Formatter)

    assert formatter.DEFAULT_MIN_LENGTH == 5
    assert formatter.DEFAULT_N_CORE == 30
    assert formatter.DEFAULT_MAX_LENGTH == 20
    assert formatter.TINY_USER_SAMPLE_SIZE == 80_000
    assert formatter._scale_dataset_prefix() == 'rvt'
    assert formatter.scale_percent() == 10
    assert formatter._parse_scale_dataset('rvt10') == 10
    assert formatter._parse_scale_dataset('rvs10') is None


def test_rvt_scale_split_matches_rvs_policy():
    formatter = _formatter(RVT10Formatter)

    assert formatter._scale_split_points(10_000) == (9_970, 1_000)


def test_rvt_samples_raw_users_before_filtering(monkeypatch):
    formatter = _formatter(RVT10Formatter)
    raw_users = pd.DataFrame(
        {
            formatter.UID_COL: [f'u{index}' for index in range(100)],
            formatter.HIS_COL: [[index] for index in range(100)],
        }
    )
    monkeypatch.setattr(RVT10Formatter, 'TINY_USER_SAMPLE_SIZE', 8)
    monkeypatch.setattr(
        RecIFVideoSmallScaleFormatter,
        '_load_raw_users',
        lambda self: raw_users,
    )

    first = formatter._load_raw_users()
    second = formatter._load_raw_users()

    assert len(first) == 8
    assert first[formatter.UID_COL].tolist() == second[formatter.UID_COL].tolist()
    assert first[formatter.UID_COL].tolist() != raw_users[formatter.UID_COL].head(8).tolist()
