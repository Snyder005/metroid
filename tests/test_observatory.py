"""Tests for metroid.observatory.Observatory."""

from astropy import units as u
from astropy.coordinates import EarthLocation
import numpy as np
import pytest

from metroid.camera import Camera
from metroid.observatory import Observatory
from metroid.photometry import PhotometricParameters, ThroughputCurve
from metroid.profiles import CircularPupil


@pytest.fixture
def camera():
    """A Camera instance."""
    bandpasses = {"lsst2023-u": ThroughputCurve.load_filter("lsst2023-u")}
    return Camera(bandpasses, 1.5 * (u.electron / u.adu), 0.2 * (u.arcsec / u.pix))


@pytest.fixture
def pupil():
    """A CircularPupil instance."""
    return CircularPupil(4.0 * u.m)


@pytest.fixture
def location():
    """A fixed EarthLocation (avoids a network site lookup)."""
    return EarthLocation.from_geodetic(lon=-70.7494 * u.deg, lat=-30.2446 * u.deg, height=2647.0 * u.m)


@pytest.fixture
def observatory(camera, pupil, location):
    """An Observatory instance."""
    return Observatory(camera, pupil, location)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_observatory_stores_components(observatory, camera, pupil, location):
    """The observatory exposes its camera, pupil, and location."""
    assert observatory.camera is camera
    assert observatory.pupil is pupil
    assert observatory.location is location


@pytest.mark.parametrize(
    "bad_camera",
    ["not a camera", None, 42],
)
def test_observatory_rejects_bad_camera(bad_camera, pupil, location):
    """A non-Camera first argument raises ValueError."""
    with pytest.raises(ValueError):
        Observatory(bad_camera, pupil, location)


def test_observatory_rejects_bad_pupil(camera, location):
    """A non-Pupil second argument raises ValueError."""
    with pytest.raises(ValueError):
        Observatory(camera, "not a pupil", location)


def test_observatory_rejects_bad_location(camera, pupil):
    """A non-EarthLocation third argument raises ValueError."""
    with pytest.raises(ValueError):
        Observatory(camera, pupil, "not a location")


# ---------------------------------------------------------------------------
# get_photo_params
# ---------------------------------------------------------------------------


def test_get_photo_params_returns_parameters(observatory):
    """get_photo_params builds PhotometricParameters from the components."""
    photo_params = observatory.get_photo_params(15.0 * u.s)
    assert isinstance(photo_params, PhotometricParameters)
    assert photo_params.exptime.value == pytest.approx(15.0)
    assert photo_params.gain.value == pytest.approx(1.5)
    assert photo_params.area.value == pytest.approx(np.pi * 4.0**2)
    assert photo_params.qe.value == pytest.approx(1.0)


def test_get_photo_params_converts_exptime_unit(observatory):
    """A non-canonical exposure-time unit is converted to seconds."""
    photo_params = observatory.get_photo_params(0.25 * u.min)
    assert photo_params.exptime.unit == u.s
    assert photo_params.exptime.value == pytest.approx(15.0)


@pytest.mark.parametrize(
    "exptime,expected_error",
    [
        (15.0, TypeError),
        (15.0 * u.m, ValueError),
    ],
)
def test_get_photo_params_invalid_exptime(observatory, exptime, expected_error):
    """A bad exposure-time type or unit raises."""
    with pytest.raises(expected_error):
        observatory.get_photo_params(exptime)
