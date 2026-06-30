import copy
from typing import Any, Self

import astropy.units as u
from speclite.filters import FilterResponse, load_filter

from .conversions import photon_flux_to_adu
from .photo_params import PhotometricParameters
from .sed import Sed
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Adu, Array, EnergyFlux, PhotonFlux, Fraction, Scalar, Wavelength


class ThroughputCurve:
    """A throughput curve representing fractional transmission as a function
    of wavelength.
    """

    @enforce_units
    def __init__(self, wavelength: Wavelength[Array], throughput: Fraction[Array], meta: dict[str, Any]):
        fr = FilterResponse(wavelength.value, throughput.value, meta)
        self.__fr = self._adopt_filter_response(fr)

    @classmethod
    def from_filter_response(cls, fr: FilterResponse) -> Self:
        """Create a `ThroughputCurve` instance from a filter response curve.

        Parameters
        ----------
        fr : `speclite.filters.FilterResponse`
            The filter response curve.

        Returns
        -------
        throughput_curve : `metroid.photometry.ThroughputCurve`
            The throughput curve initialized from the filter response curve.

        Raises
        ------
        TypeError
            Raised if ``fr`` is an invalid type.
        """
        if not isinstance(fr, FilterResponse):
            raise TypeError("fr must be 'FilterResponse'")

        throughput_curve = cls.__new__(cls)
        throughput_curve.__fr = cls._adopt_filter_response(fr)
        return throughput_curve

    @classmethod
    def load_filter(cls, name: str) -> Self:
        """Create a `ThroughputCurve` instance by loading a filter response by
        name.

        Parameters
        ----------
        name : `str`
            The filter band name.

        Returns
        -------
        throughput_curve : `metroid.photometry.ThroughputCurve`
            The throughput curve initialized from the loaded filter response.

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
    def wavelength(self) -> Wavelength[Array]:
        """The wavelength array in units of Angstroms
        (`astropy.units.Quantity`).
        """
        return self.__fr.wavelength * u.AA

    @property
    @enforce_units
    def throughput(self) -> Fraction[Array]:
        """The throughput array in dimensionless units
        (`astropy.units.Quantity`).
        """
        return self.__fr.response * u.dimensionless_unscaled

    @property
    @enforce_units
    def effective_wavelength(self) -> Wavelength[Scalar]:
        """The effective wavelength of the throughput curve in units of
        Angstroms.
        """
        return self.__fr.effective_wavelength

    @property
    @enforce_units
    def ab_zeropoint(self) -> PhotonFlux[Scalar]:
        """The AB zeropoint in units of photons per second per square meter."""
        return self.__fr.ab_zeropoint

    @enforce_units
    def calculate_photon_flux(self, brightness_spec: float | int | Sed) -> PhotonFlux[Scalar]:
        """Calculate a photon flux density.

        Parameters
        ----------
        brightness_spec : `float`, `int`, or `metroid.photometry.Sed`
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
            Raised if ``brightness_spec`` is an unsupported type.
        """
        sed = self._ensure_sed(brightness_spec)
        return self._convolve(sed, photon_weighted=True)

    @enforce_units
    def calculate_energy_flux(self, brightness_spec: float | int | Sed) -> EnergyFlux[Scalar]:
        """Calculate an energy flux density.

        Parameters
        ----------
        brightness_spec : `float`, `int`, or `metroid.photometry.Sed`
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
    def calculate_adu(
        self,
        brightness_spec: float | int | Sed,
        photo_params: PhotometricParameters,
    ) -> Adu[Scalar]:
        """Calculate the summed ADU of an observation.

        Parameters
        ----------
        brightness_spec : `float`, `int`, or `metroid.photometry.Sed`
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

    def _ensure_sed(self, brightness_spec: float | int | Sed) -> Sed:
        """Ensure the correct SED is provided an observed object.

        Parameters
        ----------
        brightness_spec : `float`, `int`, or `metroid.photometry.Sed`
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

        # ``bool`` subclasses ``int`` but is not a valid magnitude.
        elif isinstance(brightness_spec, (int, float)) and not isinstance(brightness_spec, bool):
            sed = Sed.for_ab_magnitudes()

            scale = 10 ** (-0.4 * brightness_spec)
            return Sed(sed.wavelength, sed.flambda * scale)

        else:
            raise TypeError("brightness_spec is an unsupported type")

    def _convolve(self, sed: Sed, photon_weighted: bool = True) -> u.Quantity:
        """Convolve the throughput curve with an SED.

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
    def _adopt_filter_response(fr: FilterResponse) -> FilterResponse:
        """Take private, immutable ownership of a filter response.

        ``speclite.filters.load_filter`` returns a *shared, cached*
        ``FilterResponse`` whose underlying arrays are reused across callers,
        so freezing them in place would leak read-only state into speclite's
        global cache. A ``deepcopy`` yields an independent object with
        independent arrays (and bypasses ``FilterResponse.__init__``, so it is
        not re-registered into the cache); freezing that copy is safe.

        Parameters
        ----------
        fr : `speclite.filters.FilterResponse`
            The filter response to take ownership of.

        Returns
        -------
        fr : `speclite.filters.FilterResponse`
            A private, frozen copy of the filter response.
        """
        fr = copy.deepcopy(fr)
        fr._wavelength.flags.writeable = False
        fr._response.flags.writeable = False
        return fr
