import pytest

from metroid.photometry.throughput import ThroughputCurve
from metroid.plugins.rubin import RubinBandpassProvider


def test_loads_valid():
    """Test that the loads method of RubinBandPassProvider returns correct
    result for valid cases.
    """
    provider = RubinBandpassProvider()
    bandpasses = provider.load("u")
    assert isinstance(bandpasses, dict)
    assert isinstance(bandpasses["u"], ThroughputCurve)


@pytest.mark.parametrize("name,expected_exception", [(5, TypeError), ("J", IOError)])
def test_loads_invalid(name, expected_exception):
    provider = RubinBandpassProvider()
    with pytest.raises(expected_exception):
        provider.load(name)
