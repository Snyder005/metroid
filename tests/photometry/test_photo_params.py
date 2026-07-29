"""Tests for metroid.photometry.photo_params.PhotometricParameters."""

from astropy import units as u
import pytest

from metroid.photometry.photo_params import PhotometricParameters
from metroid.utils.quantities import QuantityValidationError


def test_photo_params_stores_fields():
    """Fields are stored and converted to canonical units."""
    p = PhotometricParameters(
        exptime=15.0 * u.s,
        gain=1.5 * u.electron / u.adu,
        area=0.001 * u.km**2,
        qe=0.9 * u.electron / u.ph,
    )
    assert p.exptime.value == pytest.approx(15.0)
    assert p.gain.value == pytest.approx(1.5)
    assert p.area.unit == u.m**2
    assert p.area.value == pytest.approx(1000.0)
    assert p.qe.value == pytest.approx(0.9)


def test_photo_params_default_qe():
    """qe defaults to 1 electron per photon."""
    p = PhotometricParameters(15.0 * u.s, 1.5 * u.electron / u.adu, 1.0 * u.m**2)
    assert p.qe == 1.0 * u.electron / u.ph


def test_photo_params_is_frozen():
    """PhotometricParameters is immutable."""
    p = PhotometricParameters(15.0 * u.s, 1.5 * u.electron / u.adu, 1.0 * u.m**2)
    with pytest.raises(Exception):
        p.exptime = 20.0 * u.s  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("exptime", 15.0 * u.m),
        ("gain", 1.5 * u.s),
        ("area", 1.0 * u.m),
        ("qe", 0.9 * u.s),
    ],
)
def test_photo_params_bad_unit_raises(field, value):
    """A field with an incompatible unit raises ValueError."""
    kwargs = {
        "exptime": 15.0 * u.s,
        "gain": 1.5 * u.electron / u.adu,
        "area": 1.0 * u.m**2,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        PhotometricParameters(**kwargs)


def test_photo_params_array_field_rejected():
    """A Scalar field rejects an array value."""
    with pytest.raises(QuantityValidationError):
        PhotometricParameters([1.0, 2.0] * u.s, 1.5 * u.electron / u.adu, 1.0 * u.m**2)
