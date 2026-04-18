from metroid.plugins.registry import _PROVIDERS, register_provider, get_provider
from metroid.plugins.rubin import RubinBandpassProvider

def test_register_provider():
    register_provider('rubin', RubinBandpassProvider)
    assert _PROVIDERS['rubin'] == RubinBandpassProvider

def test_get_provider():
    register_provider('rubin', RubinBandpassProvider)
    assert isinstance(get_provider('rubin'), RubinBandpassProvider)
