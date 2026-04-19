import pytest
from astropy import units as u
import numpy as np

from metroid.cameras import Camera
from metroid.plugins.rubin import RubinBandpassProvider


@pytest.fixture
def camera():
    """A fixture returning a Camera instance."""
    gain = 1.5 * (u.electron / u.adu)
    pixel_scale = 0.2 * (u.arcsec / u.pix)
    bandpasses = RubinBandpassProvider().load("u")
    return Camera(gain, pixel_scale, bandpasses)


def test_camera_creation(camera):
    """Test the creation of a Camera instance."""
    assert camera.gain == 1.5 * (u.electron / u.adu)
    assert camera.pixel_scale == 0.2 * (u.arcsec / u.pix)
    assert camera.bands == ("u",)


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
    bandpasses = RubinBandpassProvider().load("u")
    with pytest.raises(expected_error):
        Camera(gain, pixel_scale, bandpasses)


def test_from_config():
    """Test the creation of a Camera instance from a configuration
    dictionary.
    """
    config = {"gain": 1.5, "pixel_scale": 0.2, "bands": ["u"]}
    camera = Camera.from_config(config)
    assert isinstance(camera, Camera)


def test_get_bandpass(camera):
    """Test that get_bandpass method of a Camera returns correct result for
    valid cases.
    """
    expected_bandpass = RubinBandpassProvider().load("u")["u"]
    bandpass = camera.get_bandpass("u")

    assert np.allclose(bandpass.wavelen, expected_bandpass.wavelen)
    assert np.allclose(bandpass.sb, expected_bandpass.sb)


def test_get_bandpasses(camera):
    """Test that get_bandpasses method of a Camera returns correct result for
    valid cases.
    """
    expected_bandpasses = RubinBandpassProvider().load("u")
    bandpasses = camera.get_bandpasses("u")

    assert np.allclose(bandpasses["u"].wavelen, expected_bandpasses["u"].wavelen)
    assert np.allclose(bandpasses["u"].sb, expected_bandpasses["u"].sb)
