from metroid.plugins.registry import _PROVIDERS, register_provider, get_provider
from metroid.plugins.rubin import RubinBandpassProvider


def test_register_provider():
    """Test that register_provider correctly adds provider to the registry."""
    _PROVIDERS.clear()
    register_provider("rubin", RubinBandpassProvider)
    assert _PROVIDERS["rubin"] == RubinBandpassProvider


def test_get_provider():
    """Test that get_provider returns correct result for valid cases."""
    _PROVIDERS.clear()
    register_provider("rubin", RubinBandpassProvider)
    assert isinstance(get_provider("rubin"), RubinBandpassProvider)
