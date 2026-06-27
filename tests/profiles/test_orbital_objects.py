import pytest
from astropy import units as u
from astropy.constants import G, R_earth, M_earth
import numpy as np
import galsim

from metroid.profiles.orbital_objects import CircularOrbitalObject, RectangularOrbitalObject
from metroid.profiles.pupils import CircularPupil


@pytest.fixture
def circular_object():
    """A fixture returning a CircularOrbitalObject instance."""
    return CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m)


@pytest.fixture
def rectangular_object():
    """A fixture returning a RectangularOrbitalObject instance."""
    return RectangularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 2.0 * u.m, 4.0 * u.m)


@pytest.fixture
def orbital_object(request):
    """A fixture factory returning an OrbitalObject subclass instance."""
    if request.param == "circular_object":
        return CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m)
    elif request.param == "rectangular_object":
        return RectangularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 2.0 * u.m, 4.0 * u.m)
    else:
        raise ValueError(f"Unknown pupil type: {request.param}")


@pytest.mark.parametrize(
    "orbital_object",
    ["circular_object", "rectangular_object"],
    indirect=True,
)
def test_orbital_object_creation(orbital_object):
    """Test the creation of an OrbitalObject subclass instance."""
    h = 550.0 * u.km
    theta_z = 70.0 * u.deg
    theta_r = 0.0 * u.deg

    theta_n = np.arcsin(R_earth * np.sin(theta_z) / (R_earth + h)).to(u.deg)
    d = (R_earth * np.sin(theta_z - theta_n) / np.sin(theta_n)).to(u.km)
    v_o = np.sqrt(G * M_earth / (R_earth + h)).to(u.m / u.s)
    omega_o = (v_o / (R_earth + h)).to(u.rad / u.s, equivalencies=u.dimensionless_angles())
    v_p = (v_o * np.cos(theta_n)).to(u.m / u.s)
    omega_p = (v_p / d).to(u.rad / u.s, equivalencies=u.dimensionless_angles())
    solid_angle = (orbital_object.area / d**2).to(u.sr, equivalencies=u.dimensionless_angles())

    assert orbital_object.height == h
    assert orbital_object.zenith_angle == theta_z
    assert orbital_object.rotation_angle == theta_r
    assert orbital_object.nadir_pointing is False
    assert u.isclose(orbital_object.nadir_angle, theta_n)
    assert u.isclose(orbital_object.distance, d)
    assert u.isclose(orbital_object.orbital_velocity, v_o)
    assert u.isclose(orbital_object.orbital_angular_velocity, omega_o)
    assert u.isclose(orbital_object.perpendicular_velocity, v_p)
    assert u.isclose(orbital_object.perpendicular_angular_velocity, omega_p)
    assert u.isclose(orbital_object.solid_angle, solid_angle)


@pytest.mark.parametrize(
    "orbital_object",
    ["circular_object", "rectangular_object"],
    indirect=True,
)
def test_calculate_pixel_time_valid(orbital_object):
    """Test that calculate_pixel_time method of an OrbitalObject subclass
    instance returns correct result for valid cases.
    """
    pixel_scale = 0.2 * (u.arcsec / u.pix)
    t_p = (pixel_scale / orbital_object.perpendicular_angular_velocity).to(u.s, equivalencies=[(u.pix, None)])

    assert u.isclose(orbital_object.calculate_pixel_time(pixel_scale), t_p)


@pytest.mark.parametrize(
    "orbital_object",
    ["circular_object", "rectangular_object"],
    indirect=True,
)
def test_calculate_pixel_time_invalid(orbital_object):
    """Test that calculate_pixel_time method of an OrbitalObject subclass
    instance raises proper exception for invalid cases.
    """
    with pytest.raises(ValueError):
        orbital_object.calculate_pixel_time(0.0 * (u.arcsec / u.pix))


@pytest.mark.parametrize(
    "orbital_object",
    ["circular_object", "rectangular_object"],
    indirect=True,
)
def test_get_tracked_profile(orbital_object):
    """Test that get_tracked_profile method of an OrbitalObject subclass
    instance returns correct result for valid cases.
    """
    psf = galsim.Kolmogorov(fwhm=0.7)
    pupil = CircularPupil(4.0 * u.m)

    assert isinstance(orbital_object.get_tracked_profile(psf, pupil), galsim.Convolution)


def test_circular_object_creation(circular_object):
    """Test the creation of a CircularOrbitalObject instance."""
    r = 3.0 * u.m
    area = (np.pi * r**2).to(u.m * u.m)
    expected_radius = (circular_object.radius / circular_object.distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )

    assert circular_object.radius == r
    assert circular_object.area == area
    assert isinstance(circular_object.profile, galsim.TopHat)
    assert circular_object.profile.radius == pytest.approx(expected_radius)


def test_rectangular_object_creation(rectangular_object):
    """Test the creation of a RectangularOrbitalObject instance."""
    w = 2.0 * u.m
    l = 4.0 * u.m
    area = (w * l).to(u.m * u.m)
    expected_width = (rectangular_object.width / rectangular_object.distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    expected_length = (rectangular_object.length / rectangular_object.distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )

    assert rectangular_object.width == w
    assert rectangular_object.length == l
    assert rectangular_object.area == area
    assert isinstance(rectangular_object.profile, galsim.Box)
    assert rectangular_object.profile.width == pytest.approx(expected_width)
    assert rectangular_object.profile.height == pytest.approx(expected_length)
