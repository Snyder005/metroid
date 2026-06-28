from astropy.coordinates import EarthLocation

from metroid.camera import Camera
from metroid.profiles.pupils import Pupil
from metroid.photometry.photo_params import PhotometricParameters
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Time


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

    @enforce_units
    def get_photo_params(self, exptime: Time) -> PhotometricParameters:
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
        photo_params = PhotometricParameters(
            exptime=exptime, gain=self.camera.gain, area=self.pupil.area, qe=self.camera.qe
        )
        return photo_params
