import pytest
from astropy import units as u
import galsim

from metroid.profiles.pupils import Pupil, CircularPupil, AnnularPupil


@pytest.fixture
def circular_pupil():
    """A fixture returning a CircularPupil instance."""
    return CircularPupil(4.0 * u.m)


@pytest.fixture
def annular_pupil():
    """A fixture returning an AnnularPupil instance."""
    return AnnularPupil(1.0 * u.m, 4.0 * u.m)


@pytest.fixture
def pupil(request):
    """A fixture factory returning a Pupil subclass instance."""
    if request.param == "circular_pupil":
        return CircularPupil(4.0 * u.m)
    elif request.param == "annular_pupil":
        return AnnularPupil(1.0 * u.m, 4.0 * u.m)
    else:
        raise ValueError(f"Unknown pupil type: {request.param}")


@pytest.mark.parametrize(
    "config,expected_type",
    [
        ({"type": "circular", "radius": 4.0}, CircularPupil),
        ({"type": "annular", "inner_radius": 1.0, "outer_radius": 4.0}, AnnularPupil),
    ],
)
def test_from_config(config, expected_type):
    """Test the creation of Pupil subclass instance from a configuration
    dictionary.
    """
    pupil = Pupil.from_config(config)
    assert isinstance(pupil, expected_type)


@pytest.mark.parametrize(
    "pupil",
    ["circular_pupil", "annular_pupil"],
    indirect=True,
)
@pytest.mark.parametrize(
    "distance,expected_exception",
    [
        ("not a quantity", TypeError),
        (50.0 * u.s, ValueError),
    ],
)
def test_get_profile_invalid(pupil, distance, expected_exception):
    """Test that get_profile method of a Pupil subclass instance raises proper
    exception for invalid cases.
    """
    with pytest.raises(expected_exception):
        pupil.get_profile(distance)


def test_circular_pupil_creation(circular_pupil):
    """Test the creation of a CircularPupil instance."""
    assert circular_pupil.radius == 4.0 * u.m


@pytest.mark.parametrize(
    "radius,expected_error",
    [
        (4.0, TypeError),
        (4.0 * u.s, ValueError),
    ],
)
def test_circular_pupil_creation_invalid(radius, expected_error):
    """Test that creation of a CircularPupil raises proper exception for
    invalid cases.
    """
    with pytest.raises(expected_error):
        CircularPupil(radius)


def test_circular_get_profile_valid(circular_pupil):
    """Test that get_profile method of a CircularPupil instance returns
    correct result for valid cases.
    """
    distance = 200.0 * u.km
    profile = circular_pupil.get_profile(distance)

    assert isinstance(profile, galsim.TopHat)

    expected_radius = (circular_pupil.radius / distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )

    assert profile.radius == pytest.approx(expected_radius)


def test_annular_pupil_creation(annular_pupil):
    """Test the creation of an AnnularPupil instance."""
    assert annular_pupil.inner_radius == 1.0 * u.m
    assert annular_pupil.outer_radius == 4.0 * u.m


@pytest.mark.parametrize(
    "inner_radius,outer_radius,expected_error",
    [
        (1.0, 4.0 * u.m, TypeError),
        (1.0 * u.m, 4.0, TypeError),
        (4.0 * u.m, 4.0 * u.m, ValueError),
    ],
)
def test_annular_pupil_creation_invalid(inner_radius, outer_radius, expected_error):
    """Test that creation of an AnnularPupil instance raises proper
    exception for invalid cases.
    """
    with pytest.raises(expected_error):
        AnnularPupil(inner_radius, outer_radius)


def test_annular_get_profile_valid(annular_pupil):
    """Test that get_profile method of an AnnularPupil instance returns
    correct result for valid cases.
    """
    distance = 200.0 * u.km
    profile = annular_pupil.get_profile(distance)

    assert isinstance(profile, galsim.Sum)

    obj_list = profile.obj_list
    expected_inner_radius = (annular_pupil.inner_radius / distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    expected_outer_radius = (annular_pupil.outer_radius / distance).to_value(
        u.arcsec, equivalencies=u.dimensionless_angles()
    )
    expected_flux = 1.0

    assert obj_list[0].radius == pytest.approx(expected_outer_radius)
    assert obj_list[1].original.radius == pytest.approx(expected_inner_radius)
    assert obj_list[1].flux == -((expected_inner_radius / expected_outer_radius) ** 2)
