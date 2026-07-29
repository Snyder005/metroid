"""Tests for metroid.profiles.orbital_objects."""

from astropy import units as u
from astropy.constants import G, M_earth, R_earth
import galsim
import numpy as np
import pytest

from metroid.profiles.orbital_objects import CircularOrbitalObject, RectangularOrbitalObject
from metroid.profiles.pupils import CircularPupil


@pytest.fixture
def circular_object():
    """A CircularOrbitalObject instance."""
    return CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m)


@pytest.fixture
def rectangular_object():
    """A RectangularOrbitalObject instance."""
    return RectangularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 2.0 * u.m, 4.0 * u.m)


@pytest.fixture
def orbital_object(request):
    """An OrbitalObject subclass instance selected by param."""
    if request.param == "circular_object":
        return CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m)
    elif request.param == "rectangular_object":
        return RectangularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 2.0 * u.m, 4.0 * u.m)
    raise ValueError(f"Unknown object type: {request.param}")


# ---------------------------------------------------------------------------
# Construction and stored attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_stored_attributes(orbital_object):
    """Height, zenith/rotation angle, and nadir flag are stored."""
    assert orbital_object.height == 550.0 * u.km
    assert orbital_object.zenith_angle == 70.0 * u.deg
    assert orbital_object.rotation_angle == 0.0 * u.deg
    assert orbital_object.nadir_pointing is False


def test_construction_bad_height_unit():
    """A height with an incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        CircularOrbitalObject(550.0 * u.s, 70.0 * u.deg, 3.0 * u.m)


def test_construction_bad_height_type():
    """A non-Quantity height raises TypeError."""
    with pytest.raises(TypeError):
        CircularOrbitalObject(550.0, 70.0 * u.deg, 3.0 * u.m)


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------


def test_setters_update_values(circular_object):
    """Height, zenith angle, and rotation angle setters update the value."""
    circular_object.height = 600.0 * u.km
    circular_object.zenith_angle = 45.0 * u.deg
    circular_object.rotation_angle = 30.0 * u.deg
    assert circular_object.height == 600.0 * u.km
    assert circular_object.zenith_angle == 45.0 * u.deg
    assert circular_object.rotation_angle == 30.0 * u.deg


def test_height_setter_validates_unit(circular_object):
    """The height setter rejects an incompatible unit."""
    with pytest.raises(ValueError):
        circular_object.height = 600.0 * u.s


def test_nadir_pointing_setter_rejects_non_bool(circular_object):
    """The nadir_pointing setter rejects a non-bool value."""
    with pytest.raises(ValueError):
        circular_object.nadir_pointing = "yes"


def test_nadir_pointing_setter_accepts_bool(circular_object):
    """The nadir_pointing setter accepts a bool."""
    circular_object.nadir_pointing = True
    assert circular_object.nadir_pointing is True


# ---------------------------------------------------------------------------
# Orbital mechanics (values checked against the analytic formulas)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_orbital_mechanics(orbital_object):
    """Derived orbital quantities match their analytic definitions."""
    h = 550.0 * u.km
    theta_z = 70.0 * u.deg

    theta_n = np.arcsin(R_earth * np.sin(theta_z) / (R_earth + h)).to(u.deg)
    d = (R_earth * np.sin(theta_z - theta_n) / np.sin(theta_n)).to(u.km)
    v_o = np.sqrt(G * M_earth / (R_earth + h)).to(u.m / u.s)
    omega_o = (v_o / (R_earth + h)).to(u.rad / u.s, equivalencies=u.dimensionless_angles())
    v_p = (v_o * np.cos(theta_n)).to(u.m / u.s)
    omega_p = (v_p / d).to(u.rad / u.s, equivalencies=u.dimensionless_angles())
    solid_angle = (orbital_object.area / d**2).to(u.sr, equivalencies=u.dimensionless_angles())

    assert u.isclose(orbital_object.nadir_angle, theta_n)
    assert u.isclose(orbital_object.distance, d)
    assert u.isclose(orbital_object.orbital_velocity, v_o)
    assert u.isclose(orbital_object.orbital_angular_velocity, omega_o)
    assert u.isclose(orbital_object.perpendicular_velocity, v_p)
    assert u.isclose(orbital_object.perpendicular_angular_velocity, omega_p)
    assert u.isclose(orbital_object.solid_angle, solid_angle)


def test_distance_at_zenith_equals_height():
    """At zenith angle 0 the distance equals the orbital height."""
    obj = CircularOrbitalObject(550.0 * u.km, 0.0 * u.deg, 3.0 * u.m)
    assert u.isclose(obj.distance, obj.height)


def test_nadir_angle_units(circular_object):
    """The nadir angle is an angle quantity."""
    assert circular_object.nadir_angle.unit.is_equivalent(u.deg)


# ---------------------------------------------------------------------------
# calculate_pixel_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_calculate_pixel_time(orbital_object):
    """Pixel traversal time equals pixel_scale / perpendicular angular velocity."""
    pixel_scale = 0.2 * (u.arcsec / u.pix)
    expected = (pixel_scale / orbital_object.perpendicular_angular_velocity).to(
        u.s, equivalencies=[(u.pix, None)]
    )
    assert u.isclose(orbital_object.calculate_pixel_time(pixel_scale), expected)


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_calculate_pixel_time_invalid_unit(orbital_object):
    """A pixel scale with an incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        orbital_object.calculate_pixel_time(0.2 * (u.s / u.pix))


# ---------------------------------------------------------------------------
# get_tracked_profile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_get_tracked_profile(orbital_object):
    """get_tracked_profile convolves object, defocus, and PSF."""
    psf = galsim.Kolmogorov(fwhm=0.7)
    pupil = CircularPupil(4.0 * u.m)
    assert isinstance(orbital_object.get_tracked_profile(psf, pupil), galsim.Convolution)


def test_get_tracked_profile_bad_pupil(circular_object):
    """A non-Pupil telescope argument raises TypeError."""
    psf = galsim.Kolmogorov(fwhm=0.7)
    with pytest.raises(TypeError):
        circular_object.get_tracked_profile(psf, "not a pupil")


def test_get_tracked_profile_bad_psf(circular_object):
    """A non-GSObject PSF argument raises TypeError."""
    pupil = CircularPupil(4.0 * u.m)
    with pytest.raises(TypeError):
        circular_object.get_tracked_profile("not a psf", pupil)


# ---------------------------------------------------------------------------
# CircularOrbitalObject
# ---------------------------------------------------------------------------


def test_circular_object_radius_and_area(circular_object):
    """The radius property and pi*r^2 area are correct."""
    assert circular_object.radius == 3.0 * u.m
    assert u.isclose(circular_object.area, np.pi * (3.0 * u.m) ** 2)


def test_circular_object_profile(circular_object):
    """The profile is a TopHat sized by radius/distance in arcsec."""
    expected_radius = (circular_object.radius / circular_object.distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    assert isinstance(circular_object.profile, galsim.TopHat)
    assert circular_object.profile.radius == pytest.approx(expected_radius)


def test_circular_object_nadir_pointing_projects_profile():
    """A nadir-pointing circular object returns a transformed profile."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, nadir_pointing=True)
    assert isinstance(obj.profile, galsim.GSObject)


# ---------------------------------------------------------------------------
# RectangularOrbitalObject
# ---------------------------------------------------------------------------


def test_rectangular_object_dimensions_and_area(rectangular_object):
    """The width/length properties and w*l area are correct."""
    assert rectangular_object.width == 2.0 * u.m
    assert rectangular_object.length == 4.0 * u.m
    assert u.isclose(rectangular_object.area, (2.0 * u.m) * (4.0 * u.m))


def test_rectangular_object_profile(rectangular_object):
    """The profile is a Box sized by width/length over distance in arcsec."""
    expected_width = (rectangular_object.width / rectangular_object.distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    expected_length = (rectangular_object.length / rectangular_object.distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    assert isinstance(rectangular_object.profile, galsim.Box)
    assert rectangular_object.profile.width == pytest.approx(expected_width)
    assert rectangular_object.profile.height == pytest.approx(expected_length)


def test_rectangular_object_nadir_pointing_projects_profile():
    """A nadir-pointing rectangular object returns a transformed profile."""
    obj = RectangularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 2.0 * u.m, 4.0 * u.m, nadir_pointing=True)
    assert isinstance(obj.profile, galsim.GSObject)
