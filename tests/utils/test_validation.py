import pytest
from astropy import units as u

from metroid.utils.validation import get_field_value


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
