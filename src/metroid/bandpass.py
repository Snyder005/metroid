from typing import Any, Self
import astropy.units as u
import numpy as np

from speclite.filters import FilterResponse, load_filter, load_filters
from metroid.utils.decorators import enforce_units
from metroid.utils import quantities as q
from metroid.sed import Sed


class Bandpass:
    """A throughput curve for a filter band."""

    @enforce_units
    def __init__(self, wavelength: q.Wavelength, throughput: q.Throughput, meta: dict[str, Any]):
        self._fr = FilterResponse(wavelength.value, throughput.value, meta)

    @classmethod
    def from_filter_response(cls, fr: FilterResponse) -> Self:
        """Create a `Bandpass` instance from a filter response curve.

        Parameters
        ----------
        fr : `speclite.filters.FilterResponse`
            The filter response curve.

        Returns
        -------
        bandpass : `metroid.bandpass.Bandpass`
            The bandpass initialized from the filter response curve.

        Raises
        ------
        TypeError
            Raised if ``fr`` is an invalid type.
        """
        if not isinstance(fr, FilterResponse):
            raise TypeError("must be 'FilterResponse'")

        bandpass = cls.__new__(cls)
        bandpass._fr = fr
        return bandpass

    @classmethod
    def load_filter(cls, name: str) -> Self:
        """Create a `Bandpass` instance by loading a filter response by name.

        Parameters
        ----------
        name : `str`
            The filter band name.

        Returns
        -------
        bandpass : `metroid.bandpass.Bandpass`
            The bandpass initialized from the loaded filter response.

        Raises
        ------
        ValueError
            Raised if ``name`` is invalid type or the corresponding file does
            not exist.
        RuntimeError
            Raised if filter response file is incorrectly formatted.
        """
        return cls.from_filter_response(load_filter(name))

    @property
    @enforce_units
    def wavelength(self) -> q.Wavelength:
        """The wavelength array in units of Angstrom
        (`astropy.units.Quantity`).
        """
        return self._fr.wavelength * u.AA

    @property
    @enforce_units
    def throughput(self) -> q.Throughput:
        """The throughput array in dimensionless units
        (`astropy.units.Quantity`).
        """
        return self._fr.response * u.dimensionless_unscaled

    @property
    @enforce_units
    def effective_wavelength(self) -> q.Wavelength:
        """The effective wavelength of the bandpass."""
        return self._fr.effective_wavelength

    @property
    @enforce_units
    def ab_zeropoint(self) -> q.PhotonFlux:
        """The AB magnitude zeropoint in units of photons per second per
        square meter.
        """
        return self._fr.ab_zeropoint * u.ph

    @enforce_units
    def calculate_ab_flux(self, magnitude: float) -> q.PhotonFlux:
        """Calculate the photon flux corresponding to the magnitude.

        Parameters
        ----------
        magnitude : `float`

        Returns
        -------
        photon_flux : `astropy.units.Quantity`
            The photon flux for the given magnitude in units of photons per
            second per square meter.

        """
        return self.ab_zeropoint * 10 ** (-magnitude / 2.5)

    @enforce_units
    def calculate_flux(self, sed: Sed) -> q.PhotonFlux:
        """Calculate the photon flux from an object given its SED.

        Parameters
        ----------
        sed : `metroid.sed.Sed`
            The object's SED.

        Returns
        -------
        photon_flux : `astropy.units.Quantity`
            The photon flux from the object in units of photons per second per
            square meter.
        """
        flux = self._fr.convolve_with_array(
            sed.wavelength.value, 
            sed.flambda,
            photon_weighted=True)
            interpolate=True,
            units=sed.flambda.units,
        )
        return flux * u.ph

    @enforce_units
    def calculate_ab_magnitude(self, sed: Sed) -> float:
        """Calculate the AB magnitude of an object given its SED.

        Parameters
        ----------
        sed : `metroid.sed.Sed`
            The SED of the object.

        Returns
        -------
        magnitude : `float`
            The AB magnitude of the object.
        """
        return self_fr.get_ab_magnitude(sed.flambda, sed.wavelength)
