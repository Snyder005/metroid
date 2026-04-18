from typing import Protocol

from rubin_sim.phot_utils import Bandpass #this is only for type hinting, needed?


class BandpassLoader(Protocol):
    def load(self, band: list[str]) -> dict[str, Bandpass]:
        ...
