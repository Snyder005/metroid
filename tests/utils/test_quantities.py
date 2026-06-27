from astropy import units as u
import pytest

from metroid.utils.quantities import AREA, Area, check_quantity, QuantitySpec, _extract_spec


def test_quantity_spec_creation():
    spec = QuantitySpec("area", u.m**2)

    assert spec.name == "area"
    assert spec.default == (u.m**2)
    assert spec.typical_range is None


@pytest.mark.parametrize("q", [0.010 * u.km**2, 10.0 * u.m**2])
def test_check_quantity_valid(q):
    """Test that check_quantity returns correct result for valid cases."""
    assert u.isclose(check_quantity(q, AREA), q)


@pytest.mark.parametrize(
    "q,expected_error",
    [
        (10.0, TypeError),
        (10.0 * u.m, ValueError),
        (0.0 * u.m**2, ValueError),
    ],
)
def test_check_quantity_invalid(q, expected_error):
    """Test that check_quantity raises proper exception for invalid cases."""
    with pytest.raises(expected_error):
        check_quantity(q, AREA)


@pytest.mark.parametrize(
    "annotation,expected_spec",
    {
        (Area, AREA),
        (None, None),
        (Area | None, AREA),
    },
)
def test_extract_spec(annotation, expected_spec):
    """Test that _extract_spec returns correct result for valid cases."""
    assert _extract_spec(annotation) == expected_spec
