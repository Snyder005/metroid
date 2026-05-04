import pytest
from astropy import units as u
import numpy as np

from metroid.camera import Camera
from metroid.photometry import ThroughputCurve


@pytest.fixture
def camera():
    """A fixture returning a Camera instance."""
    gain = 1.5 * (u.electron / u.adu)
    pixel_scale = 0.2 * (u.arcsec / u.pix)
    bandpasses = {"lsst2023-u": ThroughputCurve.load_filter("lsst2023-u")}
    return Camera(bandpasses, gain, pixel_scale)


def test_camera_creation(camera):
    """Test the creation of a Camera instance."""
    assert camera.qe == 1.0 * u.electron / u.ph
    assert camera.gain == 1.5 * (u.electron / u.adu)
    assert camera.pixel_scale == 0.2 * (u.arcsec / u.pix)
    assert camera.filter_names == ("lsst2023-u",)


@pytest.mark.parametrize(
    "gain,pixel_scale,expected_error",
    [
        (1.5, 0.2 * (u.arcsec / u.pix), TypeError),
        (1.5 * (u.electron / u.adu), 0.2, TypeError),
    ],
)
def test_camera_creation_invalid(gain, pixel_scale, expected_error):
    """Test that creation of a Camera raises proper exception for invalid
    cases.
    """
    bandpasses = {"lsst2023-u": ThroughputCurve.load_filter("lsst2023-u")}
    with pytest.raises(expected_error):
        Camera(bandpasses, gain, pixel_scale)


def test_get_bandpass_valid(camera):
    """Test that get_bandpass method of a Camera instance returns correct
    result for valid cases.
    """
    expected_bandpass = ThroughputCurve.load_filter("lsst2023-u")
    bandpass = camera["lsst2023-u"]

    assert np.allclose(bandpass.wavelength.value, expected_bandpass.wavelength.value)
    assert np.allclose(bandpass.throughput.value, expected_bandpass.throughput.value)


def test_get_bandpass_invalid(camera):
    """Test that get_bandpass method of a Camera instance raises proper
    exception for invalid cases.
    """
    with pytest.raises(ValueError):
        camera["unknown"]
