from importlib.metadata import entry_points

from .providers import register_provider
from metroid.utils.protocols import SupportsAvailability, BandpassProvider

_DISCOVERED = False


def load_entrypoint_plugins() -> None:
    """Load bandpass entry points."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    eps = entry_points(group="metroid.bandpass")
    for ep in eps:
        name = ep.name
        try:
            provider_cls = ep.load()
            if not isinstance(provider_cls, BandpassProvider):
                raise TypeError(f"The {name} plugin must implement protocol 'BandpassProvider'")

        except Exception as e:
            register_provider(ep.name, None, error=e)
            continue

        if isinstance(provider_cls, SupportsAvailability):
            is_available = provider_cls.is_available()
        else:
            is_available = True

        if not is_available:
            register_provider(
                name,
                provider_cls,
                error=ImportError(
                    f"The '{name}' plugin requires optional dependencies. "
                    f"Install with: pip install metroid[{name}]"
                ),
            )

        register_provider(name, provider_cls)

    _DISCOVERED = True
