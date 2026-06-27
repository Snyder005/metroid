import astropy.units as u

from metroid.photometry.conversions import energy_flux_to_radiance, photon_flux_to_adu
from metroid.photometry.photo_params import PhotometricParameters


def test_energy_flux_to_radiance():
    flux = 1000.0 * u.erg / (u.s * u.m**2)
    solid_angle = 0.001 * u.sr
    expected_radiance = (flux / solid_angle).to(u.W / u.sr / u.m**2)
    assert energy_flux_to_radiance(flux, solid_angle) == expected_radiance


def test_photon_flux_to_adu():
    photon_flux = 1000.0 * u.ph / (u.s * u.m**2)
    photo_params = PhotometricParameters(1.0 * u.s, 1.0 * u.electron / u.adu, 1.0 * u.m**2)
    expected_adu = photon_flux * u.s * u.m**2 * u.adu / u.ph
    assert photon_flux_to_adu(photon_flux, photo_params) == expected_adu
