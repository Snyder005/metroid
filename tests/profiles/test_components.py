"""Tests for metroid.profiles.components."""

from astropy import units as u
import galsim
import numpy as np
import pytest

from metroid.profiles.components import CircularComponent, RectangularComponent


@pytest.fixture
def circular_component():
    """A CircularComponent instance."""
    return CircularComponent(1.0 * u.m, reflectivity=0.5 * u.dimensionless_unscaled)


@pytest.fixture
def rectangular_component():
    """A RectangularComponent instance."""
    return RectangularComponent(2.0 * u.m, 4.0 * u.m)


# ---------------------------------------------------------------------------
# Construction, stored attributes, and unit enforcement
# ---------------------------------------------------------------------------


def test_circular_stored_attributes():
    """Radius, offset, and reflectivity are stored."""
    component = CircularComponent(
        1.0 * u.m, x0=2.0 * u.m, y0=-1.0 * u.m, reflectivity=0.5 * u.dimensionless_unscaled
    )
    assert component.radius == 1.0 * u.m
    assert component.x0 == 2.0 * u.m
    assert component.y0 == -1.0 * u.m
    assert component.reflectivity == 0.5 * u.dimensionless_unscaled


def test_component_defaults():
    """Offset defaults to the origin and reflectivity to unity."""
    component = CircularComponent(1.0 * u.m)
    assert component.x0 == 0.0 * u.m
    assert component.y0 == 0.0 * u.m
    assert component.reflectivity == 1.0 * u.dimensionless_unscaled


def test_construction_bad_radius_unit():
    """A radius with an incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        CircularComponent(1.0 * u.s)


def test_construction_bad_radius_type():
    """A non-Quantity radius raises TypeError."""
    with pytest.raises(TypeError):
        CircularComponent(1.0)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_reflectivity_out_of_range_raises(value):
    """A reflectivity outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        CircularComponent(1.0 * u.m, reflectivity=value * u.dimensionless_unscaled)


# ---------------------------------------------------------------------------
# area
# ---------------------------------------------------------------------------


def test_circular_area(circular_component):
    """The circular area is pi * radius^2."""
    assert u.isclose(circular_component.area, np.pi * (1.0 * u.m) ** 2)


def test_rectangular_area(rectangular_component):
    """The rectangular area is width * length."""
    assert u.isclose(rectangular_component.area, (2.0 * u.m) * (4.0 * u.m))


# ---------------------------------------------------------------------------
# relative_flux
# ---------------------------------------------------------------------------


def test_relative_flux_scales_with_reflectivity_and_area():
    """relative_flux equals reflectivity * area in square meters."""
    component = CircularComponent(1.0 * u.m, reflectivity=0.5 * u.dimensionless_unscaled)
    assert component.relative_flux() == pytest.approx(0.5 * np.pi)


def test_relative_flux_larger_area_brighter():
    """At equal reflectivity a larger part has larger relative flux."""
    small = RectangularComponent(1.0 * u.m, 1.0 * u.m)
    large = RectangularComponent(2.0 * u.m, 2.0 * u.m)
    assert large.relative_flux() > small.relative_flux()


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------


def test_get_profile_type_and_flux(rectangular_component):
    """get_profile returns a GSObject whose flux equals relative_flux."""
    profile = rectangular_component.get_profile(550.0 * u.km)
    assert isinstance(profile, galsim.GSObject)
    assert profile.flux == pytest.approx(rectangular_component.relative_flux())


def test_get_profile_shift_matches_offset():
    """An off-center component is shifted to the expected arcsec offset."""
    distance = 550.0 * u.km
    component = CircularComponent(1.0 * u.m, x0=3.0 * u.m, y0=-2.0 * u.m)
    profile = component.get_profile(distance)

    expected_dx = (3.0 * u.m / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
    expected_dy = (-2.0 * u.m / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
    assert profile.centroid.x == pytest.approx(expected_dx)
    assert profile.centroid.y == pytest.approx(expected_dy)


def test_get_profile_bad_distance_unit(circular_component):
    """A distance with an incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        circular_component.get_profile(550.0 * u.s)
