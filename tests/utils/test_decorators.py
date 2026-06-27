from astropy import units as u
import numpy as np

from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import GeometryLength, Area


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
