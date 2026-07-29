"""Tests for metroid.utils.decorators."""

from astropy import units as u
import numpy as np
import pytest

from metroid.utils.decorators import enforce_units, validated_dataclass
from metroid.utils.quantities import (
    Area,
    Array,
    GeometryLength,
    QuantityValidationError,
    Scalar,
    Time,
)

# ---------------------------------------------------------------------------
# enforce_units: argument and return validation
# ---------------------------------------------------------------------------


def test_enforce_units_validates_argument_unit():
    """An argument with an incompatible unit raises ValueError."""

    @enforce_units
    def f(length: GeometryLength) -> GeometryLength:
        return length

    assert f(5.0 * u.m).unit == u.m
    with pytest.raises(ValueError):
        f(5.0 * u.s)


def test_enforce_units_converts_argument_to_canonical_unit():
    """An argument is converted to the spec's canonical unit before the body."""

    @enforce_units
    def get_value(length: GeometryLength) -> float:
        return length.value

    # 5 km -> 5000 m
    assert get_value(5.0 * u.km) == pytest.approx(5000.0)


def test_enforce_units_validates_return_unit():
    """The return value is validated and converted to canonical units."""

    @enforce_units
    def area(radius: GeometryLength) -> Area:
        return np.pi * radius**2

    result = area(5.0 * u.m)
    assert result.unit == u.m**2


def test_enforce_units_converts_return_to_canonical_unit():
    """A return value in a non-canonical unit is converted."""

    @enforce_units
    def quantify(radius: float) -> GeometryLength:
        return radius * u.km

    assert quantify(0.010).unit == u.m


def test_enforce_units_untyped_argument_passed_through():
    """An argument without a quantity annotation is not validated."""

    @enforce_units
    def quantify(radius: float) -> GeometryLength:
        return radius * u.km

    # A plain float is accepted, since ``radius`` carries no QuantitySpec.
    assert quantify(2.0).value == pytest.approx(2000.0)


def test_enforce_units_none_return_annotation():
    """A ``-> None`` annotation skips return validation."""

    @enforce_units
    def f(length: GeometryLength) -> None:
        return None

    assert f(5.0 * u.m) is None


def test_enforce_units_missing_return_annotation():
    """A missing return annotation skips return validation."""

    @enforce_units
    def f(length: GeometryLength):
        return None

    assert f(5.0 * u.m) is None


def test_enforce_units_none_argument_skipped():
    """A ``None`` argument value skips validation of that parameter."""

    @enforce_units
    def f(length: GeometryLength | None = None) -> float:
        return -1.0 if length is None else length.value

    assert f() == -1.0
    assert f(5.0 * u.m) == pytest.approx(5.0)


def test_enforce_units_preserves_metadata():
    """functools.wraps preserves the wrapped function's name and docstring."""

    @enforce_units
    def my_func(length: GeometryLength) -> GeometryLength:
        """Docstring."""
        return length

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "Docstring."


# ---------------------------------------------------------------------------
# enforce_units: shape enforcement
# ---------------------------------------------------------------------------


def test_enforce_units_scalar_shape():
    """A Scalar-annotated parameter rejects arrays."""

    @enforce_units
    def needs_scalar(t: Time[Scalar]) -> Time[Scalar]:
        return t

    assert needs_scalar(5.0 * u.s).unit == u.s
    with pytest.raises(QuantityValidationError):
        needs_scalar([1.0, 2.0] * u.s)


def test_enforce_units_array_shape():
    """An Array-annotated parameter rejects scalars."""

    @enforce_units
    def needs_array(t: Time[Array]) -> Time[Array]:
        return t

    assert np.all(needs_array([1.0, 2.0] * u.s).value == [1.0, 2.0])
    with pytest.raises(QuantityValidationError):
        needs_array(5.0 * u.s)


def test_enforce_units_any_shape():
    """A bare alias imposes no shape restriction."""

    @enforce_units
    def any_shape(t: Time) -> Time:
        return t

    assert any_shape(5.0 * u.s).unit == u.s
    assert any_shape([1.0, 2.0] * u.s).unit == u.s


def test_enforce_units_keyword_and_default_arguments():
    """Defaults are applied and keyword arguments are validated."""

    @enforce_units
    def f(a: Time, b: Time = 2.0 * u.s) -> Time:
        return a + b

    assert f(1.0 * u.s).value == pytest.approx(3.0)
    assert f(a=1.0 * u.s, b=4.0 * u.s).value == pytest.approx(5.0)
    # A bad-unit default-overriding keyword is still validated.
    with pytest.raises(ValueError):
        f(1.0 * u.s, b=2.0 * u.m)


# ---------------------------------------------------------------------------
# validated_dataclass
# ---------------------------------------------------------------------------


def test_validated_dataclass_validates_and_converts():
    """A validated dataclass enforces and converts field units at init."""

    @validated_dataclass(frozen=True)
    class Params:
        exptime: Time[Scalar]
        area: Area[Scalar]

    p = Params(exptime=15.0 * u.s, area=0.001 * u.km**2)
    assert p.exptime.unit == u.s
    assert p.area.unit == u.m**2
    assert p.area.value == pytest.approx(1000.0)


def test_validated_dataclass_rejects_bad_unit():
    """A field with an incompatible unit raises ValueError at construction."""

    @validated_dataclass(frozen=True)
    class Params:
        exptime: Time[Scalar]

    with pytest.raises(ValueError):
        Params(exptime=15.0 * u.m)


def test_validated_dataclass_rejects_bad_shape():
    """A Scalar field rejects an array value at construction."""

    @validated_dataclass(frozen=True)
    class Params:
        exptime: Time[Scalar]

    with pytest.raises(QuantityValidationError):
        Params(exptime=[1.0, 2.0] * u.s)


def test_validated_dataclass_honors_frozen_kwarg():
    """The frozen dataclass keyword is forwarded, making instances immutable."""

    @validated_dataclass(frozen=True)
    class Params:
        exptime: Time[Scalar]

    p = Params(exptime=15.0 * u.s)
    with pytest.raises(Exception):
        p.exptime = 20.0 * u.s  # type: ignore[misc]
