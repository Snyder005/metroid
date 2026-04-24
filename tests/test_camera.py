import pytest
from astropy import units as u
import numpy as np

from metroid.camera import Camera
from metroid.plugins.rubin import RubinBandpassProvider


@pytest.fixture
def camera():
    """A fixture returning a Camera instance."""
    gain = 1.5 * (u.electron / u.adu)
    pixel_scale = 0.2 * (u.arcsec / u.pix)
    bandpasses = RubinBandpassProvider().load("u")
    return Camera(bandpasses, gain, pixel_scale)


def test_camera_creation(camera):
    """Test the creation of a Camera instance."""
    assert camera.qe == 1.0 * u.electron / u.ph
    assert camera.gain == 1.5 * (u.electron / u.adu)
    assert camera.pixel_scale == 0.2 * (u.arcsec / u.pix)
    assert camera.band_names == ("u",)


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
        Camera(bandpasses, gain, pixel_scale)


def test_from_config():
    """Test the creation of a Camera instance from a configuration
    dictionary.
    """
    config = {"gain": 1.5, "pixel_scale": 0.2, "bands": ["u"], "qe": 1.0}
    camera = Camera.from_config(config)
    assert isinstance(camera, Camera)


def test_get_bandpass_valid(camera):
    """Test that get_bandpass method of a Camera instance returns correct
    result for valid cases.
    """
    expected_bandpass = RubinBandpassProvider().load("u")["u"]
    bandpass = camera.get_bandpass("u")

    assert np.allclose(bandpass.wavelength.value, expected_bandpass.wavelength.value)
    assert np.allclose(bandpass.throughput.value, expected_bandpass.throughput.value)


def test_get_bandpass_invalid(camera):
    """Test that get_bandpass method of a Camera instance raises proper
    exception for invalid cases.
    """
    with pytest.raises(ValueError):
        camera.get_bandpass("unknown")


def test_get_bandpasses_valid(camera):
    """Test that get_bandpasses method of a Camera instance returns correct
    result for valid cases.
    """
    expected_bandpasses = RubinBandpassProvider().load("u")
    bandpasses = camera.bandpasses

    assert np.allclose(bandpasses["u"].wavelength.value, expected_bandpasses["u"].wavelength.value)
    assert np.allclose(bandpasses["u"].throughput.value, expected_bandpasses["u"].throughput.value)


# def test_get_throughput_valid(camera):
#    """Test that get_throughput method of a Camera instance returns correct
#    result for valid cases.
#    """
#    bandpass = RubinBandpassProvider().load("u")["u"]
#    dlambda = bandpass.wavelength[1] - bandpass.wavelength[0]
#    expected_throughput = np.sum(bandpass.sb * dlambda / bandpass.wavelen)
#
#    assert camera.get_throughput("u") == pytest.approx(expected_throughput)


# def test_get_throughput_invalid(camera):
#    """Test that get_throughput method of a Camera instance raises proper
#    exception for invalid cases.
#    """
#    with pytest.raises(ValueError):
#        camera.get_throughput("unknown")
