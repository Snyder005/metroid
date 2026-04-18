import os

from rubin_sim.phot_utils import Bandpass
from rubin_sim.data import get_data_dir


class RubinBandpassProvider:
    """A Rubin Observatory bandpass provider."""

    def load(self, *bands: str) -> dict[str, Bandpass]:
        """Load Rubin Observatory bandpasses.

        Parameters
        ----------
        *bands : `str`
            LSST Camera filter bandpass names.

        Returns
        -------
        bandpasses : `dict` [str, rubin_sim.phot_utils.Bandpass]
            A dictionary of Rubin Observatory bandpasses.

        Raises
        ------
        TypeError
            Raised if a bandpass name is an invalid type.
        """
        bandpasses = {}
        for band in bands:
            if not isinstance(band, str):
                raise TypeError("must be 'str'")
            filename = os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{band}.dat")
            bandpass = Bandpass()
            bandpass.read_throughput(filename)
            bandpasses[band] = bandpass

        return bandpasses
