from astropy import units as u
import pytest

from metroid.utils.quantities import (
    ANY_SHAPE,
    AREA,
    ARRAY,
    SCALAR,
    Area,
    Array,
    QuantitySpec,
    QuantityValidationError,
    Scalar,
    Spec,
    TIME,
    check_quantity,
    _extract_spec,
)


def test_quantity_spec_creation():
    spec = QuantitySpec("area", u.m**2)

    assert spec.name == "area"
    assert spec.default == (u.m**2)
    assert spec.equivalencies == []
    assert spec.constraints == ()


@pytest.mark.parametrize("q", [0.010 * u.km**2, 10.0 * u.m**2])
def test_check_quantity_valid(q):
    """Test that check_quantity returns correct result for valid cases."""
    assert u.isclose(check_quantity(q, AREA), q)


@pytest.mark.parametrize(
    "q,expected_error",
    [
        (10.0, TypeError),
        (10.0 * u.m, ValueError),
    ],
)
def test_check_quantity_invalid(q, expected_error):
    """Test that check_quantity raises proper exception for invalid cases."""
    with pytest.raises(expected_error):
        check_quantity(q, AREA)


def test_check_quantity_converts_to_default_unit():
    """Test that a valid quantity is converted to the spec's default unit."""
    assert check_quantity(0.010 * u.km**2, AREA).unit == (u.m**2)


def test_range_constraint_aggregates_failures():
    """Test that value-level constraints raise an aggregated error."""
    spec = Spec("gain", u.electron / u.adu).ranged(0.1, 100.0).build()

    check_quantity(1.0 * u.electron / u.adu, spec)
    with pytest.raises(QuantityValidationError):
        check_quantity(1e3 * u.electron / u.adu, spec)


def test_quantity_validation_error_is_value_error():
    """Test that QuantityValidationError is catchable as ValueError."""
    spec = Spec("time", u.s).ranged(0.0, 10.0).build()

    with pytest.raises(ValueError):
        check_quantity(100.0 * u.s, spec)


def test_check_quantity_scalar_shape():
    """Test that the SCALAR shape rejects arrays and accepts scalars."""
    check_quantity(5.0 * u.s, TIME, SCALAR)
    with pytest.raises(QuantityValidationError):
        check_quantity([1.0, 2.0] * u.s, TIME, SCALAR)


def test_check_quantity_array_shape():
    """Test that the ARRAY shape rejects scalars and accepts arrays."""
    check_quantity([1.0, 2.0] * u.s, TIME, ARRAY)
    with pytest.raises(QuantityValidationError):
        check_quantity(5.0 * u.s, TIME, ARRAY)


@pytest.mark.parametrize(
    "annotation,expected_spec,expected_shape",
    [
        (Area, AREA, ANY_SHAPE),
        (Area[Scalar], AREA, SCALAR),
        (Area[Array], AREA, ARRAY),
        (None, None, ANY_SHAPE),
        (Area[Scalar] | None, AREA, SCALAR),
    ],
)
def test_extract_spec(annotation, expected_spec, expected_shape):
    """Test that _extract_spec returns the correct (spec, shape) pair."""
    spec, shape = _extract_spec(annotation)
    assert spec == expected_spec
    assert shape is expected_shape
