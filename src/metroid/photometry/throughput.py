from typing import Any, Self, TYPE_CHECKING

import astropy.units as u
from speclite.filters import FilterResponse, load_filter

from .conversions import photon_flux_to_adu, energy_flux_to_radiance
from .photo_params import PhotometricParameters
from .sed import Sed
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Adu, EnergyFlux, PhotonFlux, Fraction, Wavelength


class ThroughputCurve:
    """A throughput curve representing fractional transmission as a function
    of wavelength.
    """

    @enforce_units
    def __init__(self, wavelength: Wavelength, throughput: Fraction, meta: dict[str, Any]):
        fr = FilterResponse(wavelength.value, throughput.value, meta)
        self.__fr = self._freeze_filter_response(fr)

    @classmethod
    def from_filter_response(cls, fr: FilterResponse) -> Self:
        """Create a `Bandpass` instance from a filter response curve.

        Parameters
        ----------
        fr : `speclite.filters.FilterResponse`
            The filter response curve.

        Returns
        -------
        bandpass : `metroid.photometry.Bandpass`
            The bandpass initialized from the filter response curve.

        Raises
        ------
        TypeError
            Raised if ``fr`` is an invalid type.
        """
        if not isinstance(fr, FilterResponse):
            raise TypeError("fr must be 'FilterResponse'")

        bandpass = cls.__new__(cls)
        bandpass._ThroughputCurve__fr = cls._freeze_filter_response(fr)
        bandpass._frozen = True
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
        bandpass : `metroid.photometry.Bandpass`
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
        """The wavelength array in units of Angstroms
        (`astropy.units.Quantity`).
        """
        return self.__fr.wavelength * u.AA

    @property
    @enforce_units
    def throughput(self) -> Fraction:
        """The throughput array in dimensionless units
        (`astropy.units.Quantity`).
        """
        return self.__fr.response * u.dimensionless_unscaled

    @property
    @enforce_units
    def effective_wavelength(self) -> Wavelength:
        """The effective wavelength of the bandpass in units of Angstroms."""
        return self.__fr.effective_wavelength

    @property
    @enforce_units
    def ab_zeropoint(self) -> PhotonFlux:
        """The AB zeropoint in units of photons per second per square meter."""
        return self.__fr.ab_zeropoint

    @enforce_units
    def calculate_photon_flux(self, brightness_spec: float | Sed) -> PhotonFlux:
        """Calculate a photon flux density.

        Parameters
        ----------
        brightness_spec : `float` or `metroid.photometry.Sed`
            The brightness specification. Can be either an AB magnitude or the
            SED of an observed object.

        Returns
        -------
        photon_flux : `astropy.units.Quantity`
            The photon flux density in units of photons per second per square
            meter.

        Raises
        ------
        TypeError
            Raised if ``brightness_spec`` is unsupported type:
        """
        sed = self._ensure_sed(brightness_spec)
        return self._convolve(sed, photon_weighted=True)

    @enforce_units
    def calculate_energy_flux(self, brightness_spec: float | Sed) -> EnergyFlux:
        """Calculate an energy flux density.

        Parameters
        ----------
        brightness_spec : `float` or `metroid.photometry.Sed`
            The brightness specification. Can be either an AB magnitude or the
            SED of an observed object.

        Returns
        -------
        energy_flux : `astropy.units.Quantity`
            The energy flux density in units of ergs per second per square
            meter.

        Raises
        ------
        TypeError
            Raised if ``brightness_spec`` is an unsupported type:
        """
        sed = self._ensure_sed(brightness_spec)
        return self._convolve(sed, photon_weighted=False)

    @enforce_units
    def calculate_adu(self, brightness_spec: float | Sed, photo_params: PhotometricParameters) -> Adu:
        """Calculate the summed ADU of an observation.

        Parameters
        ----------
        brightness_spec : `float` or `metroid.photometry.Sed`
            The brightness specification. Can be either an AB magnitude or the
            SED of an observed object.
        photo_params : `metroid.photometry.PhotometricParameters`
            The photometric parameters of the observation.

        Returns
        -------
        adu : `astropy.units.Quantity`
            The summed ADU of the observation.

        Raises
        ------
        TypeError
            Raised if ``brightness_spec`` is an unsupported type.
        """
        photon_flux = self.calculate_photon_flux(brightness_spec)
        return photon_flux_to_adu(photon_flux, photo_params)

    @enforce_units
    def calculate_ab_magnitude(self, sed: Sed) -> float:
        """Calculate the AB magnitude of an object given its SED.

        Parameters
        ----------
        sed : `metroid.photometry.Sed`
            The SED of the object.

        Returns
        -------
        magnitude : `float`
            The AB magnitude of the object.
        """
        return self.__fr.get_ab_magnitude(sed.flambda, sed.wavelength)

    def _ensure_sed(self, brightness_spec: float | Sed) -> Sed:
        """Ensure the correct SED is provided an observed object.

        Parameters
        ----------
        brightness_spec : `float` or `metroid.photometry.Sed`
            The brightness specification. Can be either an AB magnitude or the
            SED of an observed object.

        Returns
        -------
        sed : `metroid.photometry.Sed`
            The SED of the observed object.

        Raises
        ------
        TypeError
            Raised if ``brightness_spec`` is an unsupported type.
        """
        if isinstance(brightness_spec, Sed):
            return brightness_spec

        elif isinstance(brightness_spec, float):
            sed = Sed.for_ab_magnitudes()

            scale = 10 ** (-0.4 * brightness_spec)
            return Sed(sed.wavelength, sed.flambda * scale)

        else:
            raise TypeError("brightness_spec is an unsupported type")

    def _convolve(self, sed: Sed, photon_weighted: bool = True) -> u.Quantity:
        """Convolve the bandpass with an SED.

        Parameters
        ----------
        sed : `metroid.photometry.Sed`
            The SED of an observed object.
        photon_weighted : `bool`, optional
            Use photon counting weights if `True`, otherwise use unit weights.

        Returns
        -------
        result : `astropy.units.Quantity`
            The result of the convolution.
        """
        result = self.__fr.convolve_with_array(
            sed.wavelength.value,
            sed.flambda,
            photon_weighted=photon_weighted,
            interpolate=True,
            units=sed.flambda.unit,
        )
        return result

    @staticmethod
    def _freeze_filter_response(fr: FilterResponse) -> FilterResponse:
        fr._wavelength.flags.writeable = False
        fr._response.flags.writeable = False
        return fr
