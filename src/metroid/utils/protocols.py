from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
from collections.abc import Mapping

if TYPE_CHECKING:
    from metroid.bandpass import Bandpass


class BandpassProvider(Protocol):
    """A protocol class for a bandpass provider."""

    def load(self, *bands: str) -> Mapping[str, Bandpass]:
        """Load bandpasses from the provider.

        Parameters
        ----------
        *bands : `str`
            Camera filter bandpass names.

        Returns
        -------
        bandpasses : `dict` [str, rubin_sim.phot_utils.Bandpass]
            A dictionary of bandpasses.
        """
        ...
