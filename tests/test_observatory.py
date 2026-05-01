import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
import numpy as np

from metroid.observatory import Observatory
from metroid.profiles import CircularPupil
from metroid.camera import Camera
from metroid.plugins.rubin import RubinBandpassProvider
from metroid.photometry import Sed


@pytest.fixture
def observatory():
    """A fixture returning an Observatory instance."""
    bandpasses = RubinBandpassProvider().load("u")
    camera = Camera(bandpasses, 1.5 * (u.electron / u.adu), 0.2 * (u.arcsec / u.pix))
    pupil = CircularPupil(4.0 * u.m)
    location = EarthLocation.of_site("Rubin")

    return Observatory(camera, pupil, location)


def test_observatory_creation(observatory):
    """Test the creation of an Observatory instance."""
    assert isinstance(observatory.camera, Camera)
    assert isinstance(observatory.pupil, CircularPupil)
    assert isinstance(observatory.location, EarthLocation)


def test_get_photo_params(observatory):
    """Test that get_photo_params returns the correct result for valid inputs."""
    photo_params = observatory.get_photo_params(15.0 * u.s)
    assert photo_params.exptime.value == pytest.approx(15.0)
    assert photo_params.gain.value == pytest.approx(1.5)
    assert photo_params.area.value == pytest.approx(np.pi * 4.0**2)
    assert photo_params.qe.value == pytest.approx(1.0)
