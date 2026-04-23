import os

import astropy.units as u

from metroid.bandpass import Bandpass
from rubin_sim.data import get_data_dir
from rubin_sim.phot_utils import Bandpass as RubinBandpass


class RubinBandpassProvider:
    """A Rubin Observatory bandpass provider."""

    def load(self, *names: str) -> dict[str, Bandpass]:
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
        for name in names:
            if not isinstance(name, str):
                raise TypeError("must be 'str'")
            filename = os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{name}.dat")
            rubin_bp = RubinBandpass()
            rubin_bp.read_throughput(filename)

            meta = {"group_name": "rubin", "band_name": name}
            bandpass = Bandpass(rubin_bp.wavelen * u.nm, rubin_bp.sb * u.dimensionless_unscaled, meta=meta)

            bandpasses[name] = bandpass

        return bandpasses
