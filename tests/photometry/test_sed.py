"""Tests for metroid.photometry.sed.Sed."""

from astropy import units as u
import numpy as np
import pytest

from metroid.photometry.sed import Sed
from metroid.utils.quantities import QuantityValidationError


@pytest.fixture
def sed():
    """A flat Sed over a nanometer wavelength grid."""
    wavelength = np.arange(300.0, 1150.1, 0.1) * u.nm
    flambda = np.ones(len(wavelength)) * u.erg / (u.s * u.cm**2 * u.AA)
    return Sed(wavelength, flambda)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_sed_wavelength_converted_to_angstrom(sed):
    """The wavelength array is stored in Angstroms."""
    assert sed.wavelength.unit == u.AA
    assert np.allclose(sed.wavelength.value, np.arange(3000.0, 11501.0, 1.0))


def test_sed_flambda_units(sed):
    """flambda keeps its spectral-flux-density units."""
    assert sed.flambda.unit == u.erg / (u.s * u.cm**2 * u.AA)
    assert np.allclose(sed.flambda.value, 1.0)


def test_sed_length_mismatch_raises():
    """Mismatched wavelength and flambda lengths raise ValueError."""
    wavelength = np.array([300.0, 400.0, 500.0]) * u.nm
    flambda = np.array([1.0, 1.0]) * u.erg / (u.s * u.cm**2 * u.AA)
    with pytest.raises(ValueError):
        Sed(wavelength, flambda)


def test_sed_non_increasing_wavelength_raises():
    """A non-strictly-increasing wavelength array raises ValueError."""
    wavelength = np.array([300.0, 300.0, 500.0]) * u.nm
    flambda = np.ones(3) * u.erg / (u.s * u.cm**2 * u.AA)
    with pytest.raises(ValueError):
        Sed(wavelength, flambda)


def test_sed_decreasing_wavelength_raises():
    """A decreasing wavelength array raises ValueError."""
    wavelength = np.array([500.0, 400.0, 300.0]) * u.nm
    flambda = np.ones(3) * u.erg / (u.s * u.cm**2 * u.AA)
    with pytest.raises(ValueError):
        Sed(wavelength, flambda)


def test_sed_scalar_wavelength_rejected():
    """A scalar wavelength (Array shape required) raises."""
    with pytest.raises(QuantityValidationError):
        Sed(300.0 * u.nm, [1.0] * u.erg / (u.s * u.cm**2 * u.AA))


def test_sed_bad_wavelength_unit_rejected():
    """A wavelength with an incompatible unit raises ValueError."""
    wavelength = np.array([1.0, 2.0, 3.0]) * u.s
    flambda = np.ones(3) * u.erg / (u.s * u.cm**2 * u.AA)
    with pytest.raises(ValueError):
        Sed(wavelength, flambda)


# ---------------------------------------------------------------------------
# for_ab_magnitudes factory
# ---------------------------------------------------------------------------


def test_for_ab_magnitudes_returns_sed():
    """for_ab_magnitudes returns a Sed instance."""
    assert isinstance(Sed.for_ab_magnitudes(), Sed)


def test_for_ab_magnitudes_default_grid():
    """The default reference SED spans the documented wavelength grid."""
    sed = Sed.for_ab_magnitudes()
    # 300-1150 nm -> 3000-11500 Angstrom.
    assert sed.wavelength.value.min() == pytest.approx(3000.0)
    assert sed.wavelength.value.max() == pytest.approx(11500.0)


def test_for_ab_magnitudes_custom_grid():
    """Custom grid bounds and step are honored."""
    sed = Sed.for_ab_magnitudes(wl_min=400.0, wl_max=700.0, wl_step=1.0)
    assert sed.wavelength.value.min() == pytest.approx(4000.0)
    assert sed.wavelength.value.max() == pytest.approx(7000.0)


def test_for_ab_magnitudes_flambda_units():
    """The reference SED's flambda is a spectral flux density."""
    sed = Sed.for_ab_magnitudes()
    assert sed.flambda.unit == u.erg / (u.s * u.cm**2 * u.AA)


def test_for_ab_magnitudes_flambda_falls_with_wavelength():
    """The flat-AB reference flambda scales as 1/wavelength^2."""
    sed = Sed.for_ab_magnitudes()
    # f_lambda ~ 1/lambda^2, so it is monotonically decreasing.
    assert np.all(np.diff(sed.flambda.value) < 0)
