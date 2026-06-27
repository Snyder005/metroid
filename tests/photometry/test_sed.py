import pytest
from astropy import units as u
import numpy as np

from metroid.photometry.sed import Sed


@pytest.fixture
def sed():
    """A fixture returning a Sed instance."""
    wavelength = np.arange(300.0, 1150.1, 0.1) * u.nm
    flambda = np.ones(len(wavelength)) * u.erg / (u.s * u.cm**2 * u.AA)
    return Sed(wavelength, flambda)


def test_sed_creation(sed):
    """Test the creation of a Sed instance."""
    wl = np.arange(3000.0, 11501.0, 1.0)
    f = np.ones(len(wl))
    assert sed.wavelength.unit == u.AA
    assert np.allclose(sed.wavelength.value, wl)
    assert sed.flambda.unit == u.erg / (u.s * u.cm**2 * u.AA)
    assert np.allclose(sed.flambda.value, f)


def test_for_adu_magnitudes():
    """Test the creation of a Sed instance for ADU magnitudes."""
    assert isinstance(Sed.for_ab_magnitudes(), Sed)
