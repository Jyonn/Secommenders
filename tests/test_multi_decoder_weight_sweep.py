import pytest

from multi_decoder_weight_sweep import _parse_weights


def test_parse_weights_deduplicates_without_reordering():
    assert _parse_weights('0,0.5,1,0.5') == [0.0, 0.5, 1.0]


def test_parse_weights_rejects_values_outside_unit_interval():
    with pytest.raises(ValueError, match='between 0 and 1'):
        _parse_weights('0.5,1.1')
