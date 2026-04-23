import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.constants import h, c
import numpy as np

from metroid.camera import Camera
from metroid.pupils import Pupil
from metroid.sed import Sed
from metroid.photo_params import PhotometricParameters
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Time, Adu


class Observatory:
    """An astronomical observatory."""

    def __init__(self, camera: Camera, pupil: Pupil, location: EarthLocation):
        if isinstance(camera, Camera):
            self._camera = camera
        else:
            raise ValueError("must be 'Camera'")

        if isinstance(pupil, Pupil):
            self._pupil = pupil
        else:
            raise ValueError("must be 'Pupil'")

        if isinstance(location, EarthLocation):
            self._location = location
        else:
            raise ValueError("must be 'EarthLocation'")

        self._photo_params = PhotometricParameters(
            self.camera.exptime,
            self.camera.gain,
            self.pupil.area,
            qe=self.camera.qe,
        )

    @property
    def camera(self) -> Camera:
        """The observatory camera (`metroid.camera.Camera`)."""
        return self._camera

    @property
    def pupil(self) -> Pupil:
        """The observatory telescope pupil (`metroid.pupils.Pupil`)."""
        return self._pupil

    @property
    def location(self) -> EarthLocation:
        """The location of the observatory
        (`astropy.coordinates.EarthLocation`).
        """
        return self._location

    @property
    @enforce_units
    def photo_params(self) -> PhotometricParameters:
        """Create photometric parameters for an exposure.

        Parameters
        ----------
        exptime : `astropy.units.Quantity`
            The exposure time.

        Returns
        -------
        photo_params : `metroid.photo_params.PhotometricParameters`
            The photometric parameters for the exposure.

        Raises
        ------
        TypeError
            Raised if ``exptime`` is an invalid type.
        ValueError
            Raised if ``exptime`` has an invalid unit or value.
        """
        return self._photo_params

    @enforce_units
    def calculate_adu(self, name: str, brightness_spec: str | Sed) -> Adu:
        bandpass = self.camera.get_bandpass(name)
        return bandpass.calculate_adu(brightness_spec, self.photo_params)

    def calculate_adus(self, brightness_spec: str | Sed) -> dict[str, Adu]:
        return {name: self.calculate_adu(name, brightness_spec) for name in camera.band_names}
