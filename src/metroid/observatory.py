from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.constants import h, c
import numpy as np

from rubin_sim.phot_utils import PhotometricParameters, Sed
from metroid.camera import Camera
from metroid.pupils import Pupil
from metroid.utils.validation import check_quantity


class Observatory:

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
        return self._camera

    @property
    def pupil(self) -> Pupil:
        return self._pupil

    @property
    def location(self) -> EarthLocation:
        return self._location

    def get_photo_params(self, exptime: u.Quantity) -> PhotometricParameters:
        """Create photometric parameters for an exposure.

        Parameters
        ----------
        exptime : `astropy.units.Quantity`
            The exposure time.

        Returns
        -------
        photo_params : `rubin_sim.phot_utils.PhotometricParameters`
            The photometric parameters for the exposure.

        Raises
        ------
        TypeError
            Raised if ``exptime`` is an invalid type.
        ValueError
            Raised if ``exptime`` has an invalid unit or value.
        """
        photo_params = PhotometricParameters(
            exptime=check_quantity(exptime, u.s, vmin=0.0).to_value(u.s),
            nexp=1,
            gain=self.camera.gain.to_value(u.electron / u.adu),
            effarea=self.pupil.area.to_value(u.cm * u.cm),
            platescale=self.camera.pixel_scale.to_value(u.arcsec / u.pix),
        )

        return photo_params

    def calculate_adu(self, band: str, magnitude: float, exptime: u.Quantity) -> u.Quantity:
        # No magnitude type check for now
        bandpass = self.camera.get_bandpass(band)
        photo_params = self.get_photo_params(exptime)

        sed = Sed()
        sed.set_flat_sed()
        m0_adu = sed.calc_adu(bandpass, phot_params=photo_params)
        adu = m0_adu * 10.0 ** (-magnitude / 2.5)
        return adu * u.adu

    def calculate_radiant_intensity(
        self,
        band: str,
        magnitude: float,
        exptime: u.Quantity,
        distance: u.Quantity,
    ) -> float:
        adu = self.calculate_adu(band, magnitude, exptime)
        nphotons = (adu * self.camera.gain) / exptime

        bandpass = self.camera.get_bandpass(band)
        lambda0 = bandpass.calc_eff_wavelen()[0] * u.nm  # Make a method of Bandpass
        power = (nphotons * h * c / lambda0).to(u.W, equivalencies=[(u.electron, None)])

        throughput = self.camera.get_throughput(band)  # Make a method of Bandpass
        solid_angle = self.pupil.get_solid_angle(distance)
        radiant_intensity = power / throughput / solid_angle
        return radiant_intensity.to(u.W / u.sr)
