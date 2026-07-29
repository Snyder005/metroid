"""Tests for metroid.photometry.throughput.ThroughputCurve."""

from astropy import units as u
import numpy as np
import pytest
from speclite.filters import load_filter

from metroid.photometry import PhotometricParameters, Sed, ThroughputCurve


@pytest.fixture
def bandpass():
    """A ThroughputCurve built from the lsst2023-g filter arrays."""
    fr = load_filter("lsst2023-g")
    return ThroughputCurve(fr.wavelength * u.AA, fr.response * u.dimensionless_unscaled, fr.meta)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_from_arrays(bandpass):
    """A curve built from arrays exposes matching wavelength and throughput."""
    fr = load_filter("lsst2023-g")
    assert u.allclose(bandpass.wavelength, fr.wavelength * u.AA)
    assert np.allclose(bandpass.throughput.value, fr.response)


def test_from_filter_response():
    """from_filter_response builds a ThroughputCurve."""
    fr = load_filter("lsst2023-g")
    assert isinstance(ThroughputCurve.from_filter_response(fr), ThroughputCurve)


def test_from_filter_response_bad_type():
    """from_filter_response rejects a non-FilterResponse argument."""
    with pytest.raises(TypeError):
        ThroughputCurve.from_filter_response("not a filter response")


def test_load_filter():
    """load_filter builds a ThroughputCurve by name."""
    assert isinstance(ThroughputCurve.load_filter("lsst2023-g"), ThroughputCurve)


def test_init_scalar_wavelength_rejected():
    """A scalar wavelength (Array shape required) raises."""
    with pytest.raises(Exception):
        ThroughputCurve(5000.0 * u.AA, 0.5 * u.dimensionless_unscaled, {})


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_wavelength_units(bandpass):
    """The wavelength property is in Angstroms."""
    assert bandpass.wavelength.unit == u.AA


def test_throughput_dimensionless(bandpass):
    """The throughput property is dimensionless."""
    assert bandpass.throughput.unit == u.dimensionless_unscaled


def test_effective_wavelength_units(bandpass):
    """The effective wavelength is in Angstroms."""
    assert bandpass.effective_wavelength.unit == u.AA


def test_ab_zeropoint_units(bandpass):
    """The AB zeropoint is a photon flux density."""
    assert bandpass.ab_zeropoint.unit.is_equivalent(u.ph / (u.s * u.m**2))


# ---------------------------------------------------------------------------
# Flux calculations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("brightness_spec", [0.0, Sed.for_ab_magnitudes()])
def test_calculate_photon_flux_matches_zeropoint(bandpass, brightness_spec):
    """A zero-magnitude / flat-AB source yields the AB zeropoint photon flux."""
    assert u.isclose(bandpass.calculate_photon_flux(brightness_spec), bandpass.ab_zeropoint)


def test_calculate_photon_flux_scales_with_magnitude(bandpass):
    """Photon flux scales as 10**(-0.4 * mag) relative to the zeropoint."""
    flux = bandpass.calculate_photon_flux(5.0)
    assert u.isclose(flux, bandpass.ab_zeropoint * 10 ** (-0.4 * 5.0))


@pytest.mark.parametrize("brightness_spec", [0.0, Sed.for_ab_magnitudes()])
def test_calculate_energy_flux_units(bandpass, brightness_spec):
    """Energy flux is an irradiance in erg/s/m^2."""
    assert bandpass.calculate_energy_flux(brightness_spec).unit == u.erg / (u.s * u.m**2)


def test_calculate_energy_flux_scales_with_magnitude(bandpass):
    """Energy flux scales as 10**(-0.4 * mag) relative to magnitude 0."""
    ref = bandpass.calculate_energy_flux(0.0)
    flux = bandpass.calculate_energy_flux(5.0)
    assert u.isclose(flux, ref * 10 ** (-0.4 * 5.0))


@pytest.mark.parametrize("brightness_spec", [0.0, Sed.for_ab_magnitudes()])
def test_calculate_adu_units(bandpass, brightness_spec):
    """calculate_adu returns a quantity in ADU."""
    photo_params = PhotometricParameters(1.0 * u.s, 1.0 * u.electron / u.adu, 1.0 * u.m**2)
    assert bandpass.calculate_adu(brightness_spec, photo_params=photo_params).unit == u.adu


def test_calculate_adu_matches_manual_conversion(bandpass):
    """calculate_adu equals photon_flux_to_adu of the photon flux."""
    from metroid.photometry.conversions import photon_flux_to_adu

    photo_params = PhotometricParameters(10.0 * u.s, 2.0 * u.electron / u.adu, 5.0 * u.m**2)
    expected = photon_flux_to_adu(bandpass.calculate_photon_flux(18.0), photo_params)
    assert u.isclose(bandpass.calculate_adu(18.0, photo_params), expected)


def test_calculate_ab_magnitude_of_reference_is_zero(bandpass):
    """The flat-AB reference SED has AB magnitude ~0 in any band."""
    assert np.isclose(bandpass.calculate_ab_magnitude(Sed.for_ab_magnitudes()), 0.0)


def test_int_magnitude_supported(bandpass):
    """Regression for #20: int AB magnitudes agree with the float equivalent."""
    assert u.isclose(bandpass.calculate_photon_flux(20), bandpass.calculate_photon_flux(20.0))
    assert u.isclose(bandpass.calculate_energy_flux(20), bandpass.calculate_energy_flux(20.0))


def test_bool_magnitude_rejected(bandpass):
    """bool subclasses int but is not a valid magnitude."""
    with pytest.raises(TypeError):
        bandpass.calculate_photon_flux(True)


def test_unsupported_brightness_spec_rejected(bandpass):
    """A string brightness spec raises TypeError."""
    with pytest.raises(TypeError):
        bandpass.calculate_photon_flux("bright")


# ---------------------------------------------------------------------------
# Filter-response ownership
# ---------------------------------------------------------------------------


def test_wrap_does_not_freeze_speclite_cache():
    """Regression for #19: wrapping must not freeze speclite's shared arrays."""
    ThroughputCurve.load_filter("lsst2023-g")
    fr = load_filter("lsst2023-g")
    assert fr.wavelength.flags.writeable
    assert fr.response.flags.writeable


def test_owned_arrays_are_frozen(bandpass):
    """The curve's private filter-response arrays are read-only."""
    fr = bandpass._ThroughputCurve__fr
    assert not fr._wavelength.flags.writeable
    assert not fr._response.flags.writeable
