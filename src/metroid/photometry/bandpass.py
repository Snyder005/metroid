from typing import Any, Self, TYPE_CHECKING

import astropy.units as u
from speclite.filters import FilterResponse, load_filter

from .conversions import photon_flux_to_adu, energy_flux_to_radiance
from .photo_params import PhotometricParameters
from .sed import Sed
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Adu, EnergyFlux, PhotonFlux, Throughput, Wavelength


class Bandpass:
    """A throughput curve for a filter band."""

    @enforce_units
    def __init__(self, wavelength: Wavelength, throughput: Throughput, meta: dict[str, Any]):
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
            raise TypeError("fr must be 'FilterResponse'")

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
    def wavelength(self) -> Wavelength:
        """The wavelength array in units of Angstrom
        (`astropy.units.Quantity`).
        """
        return self._fr.wavelength * u.AA

    @property
    @enforce_units
    def throughput(self) -> Throughput:
        """The throughput array in dimensionless units
        (`astropy.units.Quantity`).
        """
        return self._fr.response * u.dimensionless_unscaled

    @property
    @enforce_units
    def effective_wavelength(self) -> Wavelength:
        """The effective wavelength of the bandpass."""
        return self._fr.effective_wavelength

    @property
    @enforce_units
    def ab_zeropoint(self) -> PhotonFlux:
        """The AB zeropoint in units of photons per second per square meter."""
        return self._fr.ab_zeropoint

    @enforce_units
    def calculate_photon_flux(self, brightness_spec: float | Sed) -> PhotonFlux:
        """Calculate the photon flux corresponding to the magnitude.

        Parameters
        ----------
        brightness_spec : `float` or `metroid.sed.Sed`
            The brightness specification. Can be either an AB
            magnitude or the SED of an observed object.

        Returns
        -------
        photon_flux : `astropy.units.Quantity`
            The photon flux in units of photons per second per square meter.

        Raises
        ------
        TypeError
            Raised if ``brightness_spec`` is invalid type:
        """
        sed = self._ensure_sed(brightness_spec)
        return self._convolve(sed, photon_weighted=True)

    @enforce_units
    def calculate_energy_flux(self, brightness_spec: float | Sed) -> EnergyFlux:
        sed = self._ensure_sed(brightness_spec)
        return self._convolve(sed, photon_weighted=False)

    @enforce_units
    def calculate_adu(self, brightness_spec: float | Sed, photo_params: PhotometricParameters) -> Adu:
        photon_flux = self.calculate_photon_flux(brightness_spec)
        return photon_flux_to_adu(photon_flux, photo_params)

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
        return self._fr.get_ab_magnitude(sed.flambda, sed.wavelength)

    def _ensure_sed(self, brightness_spec: float | Sed) -> Sed:
        if isinstance(brightness_spec, Sed):
            return brightness_spec

        elif isinstance(brightness_spec, float):
            sed = Sed.for_ab_magnitudes()

            scale = 10 ** (-0.4 * brightness_spec)
            return Sed(sed.wavelength, sed.flambda * scale)

        else:
            raise TypeError("brightness_spec is an unsupported type")

    def _convolve(self, sed: Sed, photon_weighted: bool = True) -> u.Quantity:
        result = self._fr.convolve_with_array(
            sed.wavelength.value,
            sed.flambda,
            photon_weighted=photon_weighted,
            interpolate=True,
            units=sed.flambda.unit,
        )
        return result  # find out units
