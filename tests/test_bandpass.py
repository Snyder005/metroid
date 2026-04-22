import pytest
from astropy import units as u
import numpy as np

from speclite.filters import load_filter
from metroid.bandpass import Bandpass
from metroid.sed import Sed


@pytest.fixture
def bandpass():
    """A fixture returning a Camera instance."""
    fr = load_filter("lsst2023-g")

    return Bandpass(fr.wavelength * u.AA, fr.response * u.dimensionless_unscaled, fr.meta)


def test_bandpass_creation(bandpass):
    """Test the creation of a Bandpass instance."""
    fr = load_filter("lsst2023-g")

    assert bandpass.wavelength.unit == u.AA
    assert np.allclose(bandpass.wavelength.value, fr.wavelength)
    assert bandpass.throughput.unit == u.dimensionless_unscaled
    assert np.allclose(bandpass.throughput.value, fr.response)
    assert bandpass.effective_wavelength.unit == u.AA
    assert bandpass.ab_zeropoint.unit == u.ph / (u.s * u.m**2)


def test_from_filter_response():
    fr = load_filter("lsst2023-g")

    assert isinstance(Bandpass.from_filter_response(fr), Bandpass)


def test_load_filter():
    assert isinstance(Bandpass.load_filter("lsst2023-g"), Bandpass)


def test_calculate_photon_flux(bandpass):
    """Test that calculate_photon_flux returns correct result for valid
    inputs.
    """
    expected_photon_flux = (bandpass._fr.ab_zeropoint * (10 ** (-5.0 / 2.5))) * u.ph
    photon_flux = bandpass.calculate_photon_flux(5.0)

    assert u.isclose(photon_flux, expected_photon_flux)
