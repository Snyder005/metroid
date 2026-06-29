from astropy import units as u
import numpy as np
import pytest

from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Array, GeometryLength, Area, QuantityValidationError, Scalar, Time


def test_enforce_units():

    @enforce_units
    def area(radius: GeometryLength) -> Area:
        return np.pi * radius**2

    assert area(5.0 * u.m).unit == (u.m**2)

    @enforce_units
    def quantify(radius: float) -> GeometryLength:
        return radius * u.km

    assert quantify(0.010).unit == u.m

    @enforce_units
    def get_value(radius: GeometryLength) -> float:
        return radius.value

    assert get_value(5.0 * u.m) == 5.0

    @enforce_units
    def null_return1(radius: GeometryLength) -> None:
        return None

    assert null_return1(5.0 * u.m) is None

    @enforce_units
    def null_return2(length: GeometryLength):
        return None

    assert null_return2(5.0 * u.m) is None


def test_enforce_units_shape():
    """Test that generic shape aliases are enforced through the decorator."""

    @enforce_units
    def needs_scalar(t: Time[Scalar]) -> Time[Scalar]:
        return t

    assert needs_scalar(5.0 * u.s).unit == u.s
    with pytest.raises(QuantityValidationError):
        needs_scalar([1.0, 2.0] * u.s)

    @enforce_units
    def needs_array(t: Time[Array]) -> Time[Array]:
        return t

    assert np.all(needs_array([1.0, 2.0] * u.s).value == [1.0, 2.0])
    with pytest.raises(QuantityValidationError):
        needs_array(5.0 * u.s)

    @enforce_units
    def any_shape(t: Time) -> Time:
        return t

    assert any_shape(5.0 * u.s).unit == u.s
    assert any_shape([1.0, 2.0] * u.s).unit == u.s
