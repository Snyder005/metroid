from importlib.metadata import entry_points

from metroid.plugins.registry import register_provider

_DISCOVERED = False


def load_entrypoint_plugins() -> None:
    """Load bandpass entry points."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    eps = entry_points(group="metroid.bandpass")
    for ep in eps:
        provider_cls = ep.load()
        register_provider(ep.name, provider_cls)

    _DISCOVERED = True
