from typing import Any
import astropy.units as u
import numpy as np

from speclite.filters import Filter
from metroid.utils.decorators import enforce_units
from metroid.utils import quantities as q
from metroid.sed import Sed


class Bandpass:

    def __init__(self, wavelength: q.Wavelength, throughput: q.Throughput, meta: dict[str, Any]):
        self._st = FilterResponse(wavelength, throughput, meta)

    @property
    @enforce_units
    def wavelength(self) -> q.Wavelength:
        return self._st.wavelength * u.AA

    @property
    @enforce_units
    def throughput(self) -> q.Throughput:
        return self._st.response * u.dimensionless_unscaled

    @property
    @enforce_units
    def effective_wavelength(self) -> q.Wavelength:
        return self._st.effective_wavelength * u.AA

    def convolve(self, sed: Sed) -> q.PhotonFluxDensity:

        photon_flux = self._st.convolve_with_array(
            sed.wavelength.value,
            sed.flambda,
            photon_weighted=True,
            interpolate=True,
            units=sed.flambda.unit,
        )
        return photon_flux * u.electron
