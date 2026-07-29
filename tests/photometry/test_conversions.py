"""Tests for metroid.photometry.conversions."""

from astropy import units as u
import pytest

from metroid.photometry.conversions import energy_flux_to_radiance, photon_flux_to_adu
from metroid.photometry.photo_params import PhotometricParameters

# ---------------------------------------------------------------------------
# energy_flux_to_radiance
# ---------------------------------------------------------------------------


def test_energy_flux_to_radiance_value():
    """Radiance is the energy flux divided by the solid angle."""
    flux = 1000.0 * u.erg / (u.s * u.m**2)
    solid_angle = 0.001 * u.sr
    expected = (flux / solid_angle).to(u.W / u.sr / u.m**2)
    assert u.isclose(energy_flux_to_radiance(flux, solid_angle), expected)


def test_energy_flux_to_radiance_units():
    """The result carries radiance units."""
    result = energy_flux_to_radiance(1000.0 * u.erg / (u.s * u.m**2), 0.001 * u.sr)
    assert result.unit.is_equivalent(u.W / (u.sr * u.m**2))


def test_energy_flux_to_radiance_bad_flux_unit():
    """A flux with an incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        energy_flux_to_radiance(1000.0 * u.s, 0.001 * u.sr)


def test_energy_flux_to_radiance_bad_flux_type():
    """A non-Quantity flux raises TypeError."""
    with pytest.raises(TypeError):
        energy_flux_to_radiance(1000.0, 0.001 * u.sr)


# ---------------------------------------------------------------------------
# photon_flux_to_adu
# ---------------------------------------------------------------------------


def test_photon_flux_to_adu_value():
    """ADU = photon_flux * exptime * qe * area / gain."""
    photon_flux = 1000.0 * u.ph / (u.s * u.m**2)
    photo_params = PhotometricParameters(2.0 * u.s, 4.0 * u.electron / u.adu, 3.0 * u.m**2)
    # qe defaults to 1 electron/ph: 1000 * 2 * 1 * 3 / 4 = 1500 adu.
    result = photon_flux_to_adu(photon_flux, photo_params)
    assert result.unit == u.adu
    assert result.value == pytest.approx(1500.0)


def test_photon_flux_to_adu_default_params():
    """With unit parameters the ADU equals the incoming photon count."""
    photon_flux = 1000.0 * u.ph / (u.s * u.m**2)
    photo_params = PhotometricParameters(1.0 * u.s, 1.0 * u.electron / u.adu, 1.0 * u.m**2)
    expected = photon_flux * u.s * u.m**2 * u.adu / u.ph
    assert u.isclose(photon_flux_to_adu(photon_flux, photo_params), expected)


def test_photon_flux_to_adu_bad_params_type():
    """A non-PhotometricParameters second argument raises TypeError."""
    photon_flux = 1000.0 * u.ph / (u.s * u.m**2)
    with pytest.raises(TypeError):
        photon_flux_to_adu(photon_flux, "not params")


def test_photon_flux_to_adu_bad_flux_unit():
    """A photon flux with an incompatible unit raises ValueError."""
    photo_params = PhotometricParameters(1.0 * u.s, 1.0 * u.electron / u.adu, 1.0 * u.m**2)
    with pytest.raises(ValueError):
        photon_flux_to_adu(1000.0 * u.s, photo_params)
