from typing import Protocol

from rubin_sim.phot_utils import Bandpass  # this is only for type hinting, needed?


class BandpassProvider(Protocol):
    """A protocol class for a bandpass provider."""

    def load(self, *bands: str) -> dict[str, Bandpass]:
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
