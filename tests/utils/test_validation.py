import pytest
from astropy import units as u

from metroid.utils.validation import check_quantity, get_field_value


@pytest.mark.parametrize(
    "q,inclusive",
    [
        (0.001 * u.km, "none"),
        (1.0 * u.m, "none"),
        (0.0 * u.m, "min"),
        (2.0 * u.m, "max"),
        (0.0 * u.m, "both"),
        (2.0 * u.m, "both"),
    ],
)
def test_check_quantity_valid(q, inclusive) -> None:
    """Test that check_quantity returns correct result for valid cases."""
    assert q == check_quantity(q, u.m, vmin=0.0, vmax=2.0, inclusive=inclusive)


@pytest.mark.parametrize(
    "q,inclusive",
    [
        (1.0 * u.s, "none"),
        (-0.1 * u.m, "none"),
        (-0.1 * u.m, "min"),
        (2.1 * u.m, "none"),
        (2.1 * u.m, "max"),
        (0.0 * u.m, "none"),
        (2.0 * u.m, "none"),
    ],
)
def test_check_quantity_invalid(q, inclusive):
    """Test that check_quantity raises proper exception for invalid cases."""
    with pytest.raises(ValueError):
        check_quantity(q, u.m, vmin=0.0, vmax=2.0, inclusive=inclusive)


def test_get_field_value_valid():
    """Test that get_field_value returns correct result for valid cases."""
    config = {"name": "value"}
    assert "value" == get_field_value(config, "name", str)


@pytest.mark.parametrize(
    "config,key,typ,expected_exception",
    [
        ({"name": "value"}, "other", str, ValueError),
        ({"name": "value"}, "name", int, TypeError),
    ],
)
def test_get_field_value_invalid(config, key, typ, expected_exception):
    """Test that get_field_value raises proper exception for invalid cases."""
    with pytest.raises(expected_exception):
        get_field_value(config, key, typ)
