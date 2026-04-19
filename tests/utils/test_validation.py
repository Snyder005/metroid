import pytest
from astropy import units as u

from metroid.utils.validation import check_quantity, get_field_value


@pytest.mark.parametrize("q", [0.010 * u.km, 10.0 * u.m])
def test_check_quantity_valid(q):
    """Test that check_quantity returns correct result for valid cases."""
    assert q.to(u.m) == check_quantity(q, "geometry_length")


@pytest.mark.parametrize(
    "q,expected_error",
    [
        (10.0, TypeError),
        (10.0 * u.s, ValueError),
        (0.0 * u.m, ValueError),
    ],
)
def test_check_quantity_invalid(q, expected_error):
    """Test that check_quantity raises proper exception for invalid cases."""
    with pytest.raises(expected_error):
        check_quantity(q, "geometry_length")


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
