"""Tests for metroid.camera.Camera."""

from astropy import units as u
import numpy as np
import pytest

from metroid.camera import Camera
from metroid.photometry import ThroughputCurve


@pytest.fixture
def bandpasses():
    """A mapping of one named ThroughputCurve."""
    return {"lsst2023-u": ThroughputCurve.load_filter("lsst2023-u")}


@pytest.fixture
def camera(bandpasses):
    """A Camera instance with default QE."""
    return Camera(bandpasses, 1.5 * (u.electron / u.adu), 0.2 * (u.arcsec / u.pix))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_camera_stores_properties(camera):
    """A Camera exposes its gain, pixel scale, and default QE."""
    assert camera.gain == 1.5 * (u.electron / u.adu)
    assert camera.pixel_scale == 0.2 * (u.arcsec / u.pix)
    assert camera.qe == 1.0 * u.electron / u.ph


def test_camera_explicit_qe(bandpasses):
    """An explicit QE is stored on the camera."""
    camera = Camera(
        bandpasses,
        1.5 * (u.electron / u.adu),
        0.2 * (u.arcsec / u.pix),
        qe=0.8 * u.electron / u.ph,
    )
    assert camera.qe == 0.8 * u.electron / u.ph


def test_camera_converts_units(bandpasses):
    """Constructor arguments are converted to canonical units."""
    camera = Camera(bandpasses, 1.5 * (u.electron / u.adu), 200.0 * (u.marcsec / u.pix))
    assert camera.pixel_scale.unit == u.arcsec / u.pix
    assert camera.pixel_scale.value == pytest.approx(0.2)


@pytest.mark.parametrize(
    "gain,pixel_scale,expected_error",
    [
        (1.5, 0.2 * (u.arcsec / u.pix), TypeError),
        (1.5 * (u.electron / u.adu), 0.2, TypeError),
        (1.5 * u.s, 0.2 * (u.arcsec / u.pix), ValueError),
        (1.5 * (u.electron / u.adu), 0.2 * u.s, ValueError),
    ],
)
def test_camera_creation_invalid(bandpasses, gain, pixel_scale, expected_error):
    """Bad gain/pixel-scale types or units raise on construction."""
    with pytest.raises(expected_error):
        Camera(bandpasses, gain, pixel_scale)


def test_camera_rejects_non_string_key():
    """A non-string bandpass key raises TypeError."""
    bandpasses = {1: ThroughputCurve.load_filter("lsst2023-u")}
    with pytest.raises(TypeError):
        Camera(bandpasses, 1.5 * (u.electron / u.adu), 0.2 * (u.arcsec / u.pix))


def test_camera_rejects_non_throughput_value():
    """A bandpass value that is not a ThroughputCurve raises TypeError."""
    bandpasses = {"lsst2023-u": "not a curve"}
    with pytest.raises(TypeError):
        Camera(bandpasses, 1.5 * (u.electron / u.adu), 0.2 * (u.arcsec / u.pix))


# ---------------------------------------------------------------------------
# Mapping interface
# ---------------------------------------------------------------------------


def test_filter_names(camera):
    """filter_names returns a tuple of the bandpass names."""
    assert camera.filter_names == ("lsst2023-u",)


def test_getitem_returns_bandpass(camera):
    """Indexing by name returns the stored ThroughputCurve."""
    bandpass = camera["lsst2023-u"]
    assert isinstance(bandpass, ThroughputCurve)
    expected = ThroughputCurve.load_filter("lsst2023-u")
    assert np.allclose(bandpass.wavelength.value, expected.wavelength.value)
    assert np.allclose(bandpass.throughput.value, expected.throughput.value)


def test_getitem_unknown_raises_value_error(camera):
    """Indexing by an unknown name raises ValueError."""
    with pytest.raises(ValueError):
        camera["unknown"]


def test_len(camera):
    """len reports the number of bandpasses."""
    assert len(camera) == 1


def test_iter(camera):
    """Iterating a Camera yields its bandpass names."""
    assert list(camera) == ["lsst2023-u"]


def test_multiple_bandpasses():
    """A Camera holds and reports multiple named bandpasses."""
    bandpasses = {
        "lsst2023-u": ThroughputCurve.load_filter("lsst2023-u"),
        "lsst2023-g": ThroughputCurve.load_filter("lsst2023-g"),
    }
    camera = Camera(bandpasses, 1.5 * (u.electron / u.adu), 0.2 * (u.arcsec / u.pix))
    assert len(camera) == 2
    assert set(camera.filter_names) == {"lsst2023-u", "lsst2023-g"}


def test_bandpasses_are_read_only(camera):
    """The internal bandpass mapping does not support item assignment."""
    with pytest.raises(TypeError):
        camera._bandpasses["x"] = None  # type: ignore[index]
