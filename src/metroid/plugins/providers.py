from __future__ import annotations

from typing import TypedDict

from metroid.utils.protocols import BandpassProvider


class _ProviderEntry(TypedDict):
    provider: type[BandpassProvider] | None
    error: Exception | None


_PROVIDERS: dict[str, _ProviderEntry] = {}
"""The registry of bandpass providers."""


def register_provider(
    name: str, provider: type[BandpassProvider] | None, error: Exception | None = None
) -> None:
    """Register a bandpass provider.

    Parameters
    ----------
    name : `str`
        The name of the bandpass provider.
    provider : `metroid.utils.protocols.BandpassProvider`
        An object implementing the BandpassProvider protocol. Must define a
        ``load()`` method returning bandpasses.
    """
    _PROVIDERS[name] = _ProviderEntry(provider=provider, error=error)


def all_providers() -> dict[str, type[BandpassProvider]]:
    """Get a dictionary of all providers.

    Returns
    -------
    providers : `dict` (str, metroid.utils.protocols.BandpassProvider`
    """
    return dict(_PROVIDERS)


def get_provider(name: str) -> type[BandpassProvider]:
    """Get a bandpass provider class from the registry.

    Parameters
    ----------
    name : `str`
        The name of the bandpass provider.

    Returns
    -------
    provider : `metroid.utils.protocols.BandpassProvider`
        An object implementing the BandpassProvider protocol. Must define a
        ``load()`` method returning bandpasses.

    Raises
    ------
    ValueError
        Raised if the bandpass plugin is unknown.
    """
    try:
        entry = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown bandpass plugin: {name}")

    provider = entry["provider"]
    error = entry["error"]

    if error is not None:
        raise RuntimeError(f"Bandpass plugin '{name}' is not available") from error

    if provider is None:
        raise RuntimeError(f"Bandpass plugin '{name}' is not properly registered")

    print(entry, provider, error)
    return provider


def create_provider(name: str, *args: object, **kwargs: object) -> BandpassProvider:
    """Create a bandpass provider instance from the registry.

    Parameters
    ----------
    name : `str`
        The name of the bandpass provider.
    *args
        Variable length positional arguments of the bandpass provider class.
    **kwargs
        Arbitrary keyword arguments of the bandpass provider class.

    Returns
    -------
    bandpass_provider : `metroid.utils.protocols.BandpassProvider`
        An instance of an object implementing the BandpassProvider protocol.
        Must define a ``load()`` method returning bandpasses.
    """
    return get_provider(name)(*args, **kwargs)
