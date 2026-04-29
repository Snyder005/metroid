from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import astropy.units as u

from metroid.photometry import Bandpass

if TYPE_CHECKING:
    try:
        from rubin_sim.phot_utils import Bandpass as RubinBandpass
    except ImportError:
        RubinBandpass = Any


class RubinBandpassProvider:
    """A Rubin Observatory bandpass provider."""

    def load(self, *names: str) -> dict[str, Bandpass]:
        """Load Rubin Observatory bandpasses.

        Parameters
        ----------
        *names : `str`
            LSST Camera filter bandpass names.

        Returns
        -------
        bandpasses : `dict` [str, metroid.bandpass.Bandpass]
            A dictionary of Rubin Observatory bandpasses.

        Raises
        ------
        IOError
            Raised if the throughput file does not exist.
        TypeError
            Raised if a bandpass name is an invalid type.
        """
        get_data_dir, RubinBandpass = self._require_rubin()
        bandpasses = {}
        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"The bandpass name {name} must be 'str'")
            filename = os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{name}.dat")
            rubin_bp = RubinBandpass()
            rubin_bp.read_throughput(filename)

            meta = {"group_name": "rubin", "band_name": name}
            bandpass = Bandpass(rubin_bp.wavelen * u.nm, rubin_bp.sb * u.dimensionless_unscaled, meta=meta)

            bandpasses[name] = bandpass

        return bandpasses

    def _require_rubin(self) -> tuple[Callable[[], str], type[RubinBandpass]]:
        """Get required imports from the `rubin_sim` library."""
        try:
            return self._rubin

        except AttributeError:
            try:
                from rubin_sim.data import get_data_dir
                from rubin_sim.phot_utils import Bandpass as RubinBandpass
            except ImportError as e:
                raise ImportError(
                    "The 'rubin' plugin requires the optional dependency 'rubin_sim'. "
                    "Install it with: pip install metroid[rubin]"
                ) from e

            self._rubin = (get_data_dir, RubinBandpass)

        return self._rubin

    @staticmethod
    def is_available() -> bool:
        """Check if the provider is available for initialization."""
        try:
            import rubin_sim  # noqa: F401

            return True

        except ImportError:
            return False
