"""Tests for metroid.profiles.pupils."""

from astropy import units as u
import galsim
import numpy as np
import pytest

from metroid.profiles.pupils import AnnularPupil, CircularPupil, Pupil


@pytest.fixture
def circular_pupil():
    """A CircularPupil instance."""
    return CircularPupil(4.0 * u.m)


@pytest.fixture
def annular_pupil():
    """An AnnularPupil instance."""
    return AnnularPupil(1.0 * u.m, 4.0 * u.m)


@pytest.fixture
def pupil(request):
    """A Pupil subclass instance selected by param."""
    if request.param == "circular_pupil":
        return CircularPupil(4.0 * u.m)
    elif request.param == "annular_pupil":
        return AnnularPupil(1.0 * u.m, 4.0 * u.m)
    raise ValueError(f"Unknown pupil type: {request.param}")


# ---------------------------------------------------------------------------
# from_config registry dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config,expected_type",
    [
        ({"type": "circular", "radius": 4.0}, CircularPupil),
        ({"type": "annular", "inner_radius": 1.0, "outer_radius": 4.0}, AnnularPupil),
    ],
)
def test_from_config_dispatch(config, expected_type):
    """from_config builds the subclass named by the 'type' field."""
    pupil = Pupil.from_config(config)
    assert isinstance(pupil, expected_type)


def test_from_config_values(circular_pupil):
    """from_config passes field values through to the constructor."""
    pupil = Pupil.from_config({"type": "circular", "radius": 4.0})
    assert pupil.radius == 4.0 * u.m


def test_from_config_missing_type_raises():
    """A config lacking 'type' raises ValueError."""
    with pytest.raises(ValueError):
        Pupil.from_config({"radius": 4.0})


def test_from_config_unknown_type_raises():
    """An unknown pupil type raises ValueError."""
    with pytest.raises(ValueError):
        Pupil.from_config({"type": "triangular", "radius": 4.0})


def test_from_config_does_not_mutate_input():
    """from_config copies the config and leaves the caller's dict intact."""
    config = {"type": "circular", "radius": 4.0}
    Pupil.from_config(config)
    assert config == {"type": "circular", "radius": 4.0}


def test_from_config_missing_field_raises():
    """A config missing a required subclass field raises ValueError."""
    with pytest.raises(ValueError):
        Pupil.from_config({"type": "circular"})


# ---------------------------------------------------------------------------
# CircularPupil
# ---------------------------------------------------------------------------


def test_circular_pupil_radius(circular_pupil):
    """The radius property returns the constructed value."""
    assert circular_pupil.radius == 4.0 * u.m


def test_circular_pupil_area(circular_pupil):
    """The area is pi * radius^2."""
    assert u.isclose(circular_pupil.area, np.pi * (4.0 * u.m) ** 2)


@pytest.mark.parametrize(
    "radius,expected_error",
    [
        (4.0, TypeError),
        (4.0 * u.s, ValueError),
    ],
)
def test_circular_pupil_invalid(radius, expected_error):
    """A bad radius type or unit raises on construction."""
    with pytest.raises(expected_error):
        CircularPupil(radius)


def test_circular_get_profile(circular_pupil):
    """get_profile returns a TopHat sized by radius/distance in arcsec."""
    distance = 200.0 * u.km
    profile = circular_pupil.get_profile(distance)
    assert isinstance(profile, galsim.TopHat)
    expected_radius = (circular_pupil.radius / distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    assert profile.radius == pytest.approx(expected_radius)


# ---------------------------------------------------------------------------
# AnnularPupil
# ---------------------------------------------------------------------------


def test_annular_pupil_radii(annular_pupil):
    """The inner and outer radius properties return constructed values."""
    assert annular_pupil.inner_radius == 1.0 * u.m
    assert annular_pupil.outer_radius == 4.0 * u.m


def test_annular_pupil_area(annular_pupil):
    """The area is pi * (outer^2 - inner^2)."""
    expected = np.pi * ((4.0 * u.m) ** 2 - (1.0 * u.m) ** 2)
    assert u.isclose(annular_pupil.area, expected)


@pytest.mark.parametrize(
    "inner_radius,outer_radius,expected_error",
    [
        (1.0, 4.0 * u.m, TypeError),
        (1.0 * u.m, 4.0, TypeError),
        (4.0 * u.m, 4.0 * u.m, ValueError),
        (5.0 * u.m, 4.0 * u.m, ValueError),
    ],
)
def test_annular_pupil_invalid(inner_radius, outer_radius, expected_error):
    """Bad radius types, or outer <= inner, raise on construction."""
    with pytest.raises(expected_error):
        AnnularPupil(inner_radius, outer_radius)


def test_annular_get_profile(annular_pupil):
    """get_profile returns a difference of TopHats scaled by (r_i/r_o)^2."""
    distance = 200.0 * u.km
    profile = annular_pupil.get_profile(distance)
    assert isinstance(profile, galsim.Sum)

    obj_list = profile.obj_list
    expected_inner = (annular_pupil.inner_radius / distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    expected_outer = (annular_pupil.outer_radius / distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    assert obj_list[0].radius == pytest.approx(expected_outer)
    assert obj_list[1].original.radius == pytest.approx(expected_inner)
    assert obj_list[1].flux == pytest.approx(-((expected_inner / expected_outer) ** 2))


# ---------------------------------------------------------------------------
# Shared get_profile validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pupil", ["circular_pupil", "annular_pupil"], indirect=True)
@pytest.mark.parametrize(
    "distance,expected_error",
    [
        ("not a quantity", TypeError),
        (50.0 * u.s, ValueError),
    ],
)
def test_get_profile_invalid(pupil, distance, expected_error):
    """A bad distance type or unit raises."""
    with pytest.raises(expected_error):
        pupil.get_profile(distance)
