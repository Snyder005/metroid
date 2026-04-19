import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
import numpy as np

from metroid.observatory import Observatory
from metroid.pupils import CircularPupil
from metroid.camera import Camera
from metroid.plugins.rubin import RubinBandpassProvider


@pytest.fixture
def observatory():
    """A fixture returning an Observatory instance."""
    bandpasses = RubinBandpassProvider().load("u")
    camera = Camera(1.5 * (u.electron / u.adu), 0.2 * (u.arcsec / u.pix), bandpasses)
    pupil = CircularPupil(4.0 * u.m)
    location = EarthLocation.of_site("Rubin")

    return Observatory(camera, pupil, location)


def test_observatory_creation(observatory):
    """Test the creation of an Observatory instance."""
    assert isinstance(observatory.camera, Camera)
    assert isinstance(observatory.pupil, CircularPupil)
    assert isinstance(observatory.location, EarthLocation)


def test_get_photo_params(observatory):
    photo_params = observatory.get_photo_params(15.0 * u.s)

    assert photo_params.exptime == pytest.approx(15.0)
    assert photo_params.nexp == 1
    assert photo_params.gain == pytest.approx(1.5)
    assert photo_params.effarea == pytest.approx(np.pi * (4.0 * 100.0) ** 2)
    assert photo_params.platescale == pytest.approx(0.2)


def test_calculate_adu(observatory):
    adu = observatory.calculate_adu("u", 5.0, 15.0 * u.s)

    assert isinstance(adu, u.Quantity)


def test_calculate_radiant_intensity(observatory):
    radiant_intensity = observatory.calculate_radiant_intensity("u", 5.0, 15.0 * u.s, 500.0 * u.km)

    assert isinstance(radiant_intensity, u.Quantity)
