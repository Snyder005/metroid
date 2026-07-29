"""Tests for metroid.utils.validation."""

import pytest

from metroid.utils.validation import get_field_value


def test_get_field_value_returns_value():
    """A present field of the requested type is returned."""
    config = {"name": "value"}
    assert get_field_value(config, "name", str) == "value"


def test_get_field_value_returns_numeric_value():
    """A numeric field is returned when the type matches."""
    config = {"radius": 4.0}
    assert get_field_value(config, "radius", float) == 4.0


def test_get_field_value_missing_field_raises_value_error():
    """A missing field raises ValueError naming the field."""
    with pytest.raises(ValueError):
        get_field_value({"name": "value"}, "other", str)


def test_get_field_value_wrong_type_raises_type_error():
    """A value of the wrong type raises TypeError."""
    with pytest.raises(TypeError):
        get_field_value({"name": "value"}, "name", int)


def test_get_field_value_non_string_name_raises_type_error():
    """A non-string field name raises TypeError."""
    with pytest.raises(TypeError):
        get_field_value({"name": "value"}, 123, str)  # type: ignore[arg-type]


def test_get_field_value_uses_isinstance_semantics():
    """Type checking uses isinstance, so a bool satisfies an int request."""
    assert get_field_value({"n": True}, "n", int) is True
