from typing import Self

from astropy import units as u
from astropy.constants import c
import numpy as np

from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Array, Wavelength, SpectralFluxDensity


class Sed:
    """A spectral energy distribution function."""

    _ab_constant = 3631.0 * c * u.Jy
    """Constant spectral flux density for zero magnitude AB source."""

    @enforce_units
    def __init__(self, wavelength: Wavelength[Array], flambda: SpectralFluxDensity[Array]):
        if len(wavelength.value) != len(flambda.value):
            raise ValueError("wavelength and flambda arrays must have same length.")

        if not np.all(np.diff(wavelength.value) > 0):
            raise ValueError("wavelength values must be strictly increasing.")

        self._wavelength = wavelength
        self._flambda = flambda

    @classmethod
    def for_ab_magnitudes(cls, wl_min: float = 300.0, wl_max: float = 1150.0, wl_step: float = 0.1) -> Self:
        """Create a `Sed` instance for AB magnitude calculations.

        Parameters
        ----------
        wl_min : `float`
            The minimum wavelength, in nanometers.
        wl_max : `float`
            The maximum wavelength, in nanometers.
        wl_step : `float`
            The wavelength step size, in nanometers.

        Returns
        -------
        sed : `Sed`
            An instance of `Sed` initialized for AB magnitude calculations.
        """
        wavelength = np.arange(wl_min, wl_max + wl_step, wl_step) * u.nm
        flambda = (cls._ab_constant / wavelength**2).to(u.erg / (u.s * u.cm**2 * u.AA))

        return cls(wavelength, flambda)

    @property
    @enforce_units
    def wavelength(self) -> Wavelength[Array]:
        """The SED wavelength array in units of Angstrom
        (`astropy.units.Quantity`).
        """
        return self._wavelength

    @property
    @enforce_units
    def flambda(self) -> SpectralFluxDensity[Array]:
        """The SED flux density array in ergs per second per square meters per
        Angstrom (`astropy.units.Quantity`).
        """
        return self._flambda
