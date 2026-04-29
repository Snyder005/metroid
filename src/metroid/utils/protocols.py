from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING
from collections.abc import Mapping

if TYPE_CHECKING:
    from metroid.photometry import Bandpass


@runtime_checkable
class BandpassProvider(Protocol):
    """A protocol for a plug-in class that is a bandpass provider."""

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


@runtime_checkable
class SupportsAvailability(Protocol):
    """A protocol for a plug-in class that supports availability checking."""

    def is_available() -> bool:
        """Check if the class is available for initialization."""
        ...
