from __future__ import annotations

from metroid.utils.protocols import BandpassProvider

_PROVIDERS: dict[str, type[BandpassProvider]] = {}
"""The registry of bandpass providers."""


def register_provider(name: str, provider: type[BandpassProvider]) -> None:
    """Register a bandpass provider.

    Parameters
    ----------
    name : `str`
        The name of the bandpass provider.
    provider : `metroid.utils.protocols.BandpassProvider`
        An object implementing the BandpassProvider protocol. Must define a
        ``load()`` method returning bandpasses.
    """
    _PROVIDERS[name] = provider


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
        return _PROVIDERS[name]

    except KeyError:
        raise ValueError(f"unknown bandpass plugin: {name}")


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
