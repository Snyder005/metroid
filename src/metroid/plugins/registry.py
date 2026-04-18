from metroid.utils.protocols import BandpassProvider

_PROVIDERS: dict[str, BandpassProvider] = {}


def register_provider(name: str, provider: BandpassProvider) -> None:
    _PROVIDERS[name] = provider


def get_provider(name: str) -> BandpassProvider:
    try:
        return _PROVIDERS[name]

    except KeyError:
        raise ValueError(f"unknown bandpass plugin: {name}")
