from typing import Type

from metroid.utils.protocols import BandpassProvider

_PROVIDERS: dict[str, Type[BandpassProvider]] = {}
"""The registry of bandpass providers."""


def register_provider(name: str, provider: Type[BandpassProvider]) -> None:
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


def get_provider(name: str) -> Type[BandpassProvider]:
    """Get a bandpass provider.

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
        return _PROVIDERS[name]()

    except KeyError:
        raise ValueError(f"unknown bandpass plugin: {name}")
