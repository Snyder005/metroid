from metroid.plugins.discovery import load_entrypoint_plugins
from metroid.plugins.providers import _PROVIDERS
from metroid.plugins.rubin import RubinBandpassProvider


def test_load_entrypoint_plugins():
    """Test that load_entrypoint_plugins correctly sets provider."""
    _PROVIDERS.clear()
    load_entrypoint_plugins()
    assert _PROVIDERS["rubin"]["provider"] == RubinBandpassProvider
