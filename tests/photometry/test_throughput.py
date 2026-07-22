import pytest
from astropy import units as u
import numpy as np

from speclite.filters import load_filter
from metroid.photometry import ThroughputCurve, Sed, PhotometricParameters


@pytest.fixture
def bandpass():
    """A fixture returning a Camera instance."""
    fr = load_filter("lsst2023-g")

    return ThroughputCurve(fr.wavelength * u.AA, fr.response * u.dimensionless_unscaled, fr.meta)


def test_bandpass_creation(bandpass):
    """Test the creation of a Bandpass instance."""
    fr = load_filter("lsst2023-g")

    assert u.allclose(bandpass.wavelength, fr.wavelength * u.AA)
    assert np.allclose(bandpass.throughput, fr.response * u.dimensionless_unscaled)
    assert bandpass.effective_wavelength.unit == u.AA
    assert bandpass.ab_zeropoint.unit == u.ph / (u.s * u.m**2)


def test_from_filter_response():
    fr = load_filter("lsst2023-g")

    assert isinstance(ThroughputCurve.from_filter_response(fr), ThroughputCurve)


def test_load_filter():
    assert isinstance(ThroughputCurve.load_filter("lsst2023-g"), ThroughputCurve)


@pytest.mark.parametrize("brightness_spec", [0.0, Sed.for_ab_magnitudes()])
def test_calculate_photon_flux(bandpass, brightness_spec):
    """Test that calculate_photon_flux returns correct result for valid
    inputs.
    """
    assert u.isclose(bandpass.calculate_photon_flux(brightness_spec), bandpass.ab_zeropoint)


@pytest.mark.parametrize("brightness_spec", [0.0, Sed.for_ab_magnitudes()])
def test_calculate_energy_flux(bandpass, brightness_spec):
    """Test that calculate_energy_flux returns correct result for valid
    inputs.
    """
    assert bandpass.calculate_energy_flux(brightness_spec).unit == u.erg / (u.s * u.m**2)


@pytest.mark.parametrize("brightness_spec", [0.0, Sed.for_ab_magnitudes()])
def test_calculate_adu(bandpass, brightness_spec):
    photo_params = PhotometricParameters(1.0 * u.s, 1.0 * u.electron / u.adu, 1.0 * u.m**2)
    assert bandpass.calculate_adu(brightness_spec, photo_params=photo_params).unit == u.adu


def test_calculate_ab_magnitude(bandpass):
    sed = Sed.for_ab_magnitudes()

    assert np.isclose(bandpass.calculate_ab_magnitude(sed), 0.0)


def test_int_magnitude_supported(bandpass):
    """Regression for #20: int AB magnitudes must be accepted and agree with
    the equivalent float.
    """
    assert u.isclose(bandpass.calculate_photon_flux(20), bandpass.calculate_photon_flux(20.0))
    assert u.isclose(bandpass.calculate_energy_flux(20), bandpass.calculate_energy_flux(20.0))


def test_bool_magnitude_rejected(bandpass):
    """``bool`` subclasses ``int`` but is not a valid magnitude."""
    with pytest.raises(TypeError):
        bandpass.calculate_photon_flux(True)


def test_wrap_does_not_freeze_speclite_cache():
    """Regression for #19: wrapping a filter must not freeze speclite's shared
    cached arrays.
    """
    ThroughputCurve.load_filter("lsst2023-g")
    fr = load_filter("lsst2023-g")

    assert fr.wavelength.flags.writeable
    assert fr.response.flags.writeable
