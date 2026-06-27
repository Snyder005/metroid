from typing import Self

from astropy import units as u
from astropy.constants import c
import numpy as np
from speclite.filters import _ab_constant, validate_wavelength_array

from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Wavelength, SpectralFluxDensity


class Sed:
    """A spectral energy distribution function."""

    @enforce_units
    def __init__(self, wavelength: Wavelength, flambda: SpectralFluxDensity):
        if len(wavelength.value) != len(flambda.value):
            raise ValueError("wavelength and flambda arrays must have same length.")

        self._wavelength = validate_wavelength_array(wavelength, min_length=2) * u.AA
        self._flambda = flambda

    @classmethod
    def for_ab_magnitudes(cls, wl_min: float = 300.0, wl_max: float = 1150.0, wl_step: float = 0.1) -> Self:
        """Create a `Sed` instance for AB magnitude calculations.

        Parameters
        ----------
        wl_min : `float`
            The minimum wavelength.
        wl_max : `float`
            Maximum wavelength value.
        wl_step : `float`
            The wavelength increment value.

        Returns
        -------
        sed : `Sed`
            An instance of `Sed` initialized for AB magnitude calcultions.
        """
        wavelength = np.arange(wl_min, wl_max + wl_step, wl_step) * u.nm
        flambda = (_ab_constant / wavelength**2).to(u.erg / (u.s * u.cm**2 * u.AA))

        return cls(wavelength, flambda)

    @property
    @enforce_units
    def wavelength(self) -> Wavelength:
        """The SED wavelength array in units of Angstrom
        (`astropy.units.Quantity`).
        """
        return self._wavelength

    @property
    @enforce_units
    def flambda(self) -> SpectralFluxDensity:
        """The SED flux density array in ergs per second per square meters per
        Angstrom (`astropy.units.Quantity`).
        """
        return self._flambda
