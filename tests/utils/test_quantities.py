"""Tests for metroid.utils.quantities."""

from astropy import units as u
import numpy as np
import pytest

from metroid.utils.quantities import (
    ANY_SHAPE,
    AREA,
    ARRAY,
    SCALAR,
    TIME,
    AnyShape,
    Area,
    Array,
    Constraint,
    Finite,
    QuantitySpec,
    QuantityValidationError,
    Range,
    Scalar,
    ShapeKind,
    SOLID_ANGLE,
    Spec,
    Time,
    check_quantity,
    _extract_spec,
    _spec_from_annotated,
)

# ---------------------------------------------------------------------------
# QuantityValidationError
# ---------------------------------------------------------------------------


def test_validation_error_is_value_error():
    """QuantityValidationError subclasses ValueError."""
    assert issubclass(QuantityValidationError, ValueError)


def test_validation_error_records_name_and_problems():
    """The error stores name and problems and formats a joined message."""
    err = QuantityValidationError("time", ["too big", "not finite"])
    assert err.name == "time"
    assert err.problems == ["too big", "not finite"]
    assert "time" in str(err)
    assert "too big" in str(err)
    assert "not finite" in str(err)


# ---------------------------------------------------------------------------
# Constraints: Range and Finite
# ---------------------------------------------------------------------------


def test_range_within_bounds_returns_none():
    """Range.check returns None when all values are inside the bounds."""
    assert Range(0.0, 10.0).check(5.0 * u.s, "time") is None


def test_range_inclusive_bounds():
    """Range bounds are inclusive at both ends."""
    r = Range(0.0, 10.0)
    assert r.check(0.0 * u.s, "time") is None
    assert r.check(10.0 * u.s, "time") is None


def test_range_out_of_bounds_returns_message():
    """Range.check returns a message when a value is outside the bounds."""
    msg = Range(0.0, 10.0).check(11.0 * u.s, "time")
    assert msg is not None
    assert "range" in msg


def test_range_checks_every_element_of_array():
    """Range.check fails if any element of an array is out of bounds."""
    r = Range(0.0, 10.0)
    assert r.check([1.0, 2.0, 3.0] * u.s, "time") is None
    assert r.check([1.0, 20.0] * u.s, "time") is not None


def test_finite_accepts_finite_values():
    """Finite.check returns None for finite scalars and arrays."""
    assert Finite().check(5.0 * u.s, "time") is None
    assert Finite().check([1.0, 2.0] * u.s, "time") is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_finite_rejects_non_finite(bad):
    """Finite.check returns a message for NaN or infinite values."""
    msg = Finite().check(bad * u.s, "time")
    assert msg is not None
    assert "non-finite" in msg


def test_range_and_finite_satisfy_constraint_protocol():
    """Range and Finite are runtime-checkable Constraint instances."""
    assert isinstance(Range(0.0, 1.0), Constraint)
    assert isinstance(Finite(), Constraint)


# ---------------------------------------------------------------------------
# Shape kinds
# ---------------------------------------------------------------------------


def test_scalar_shape_accepts_scalar_rejects_array():
    """SCALAR passes scalars and fails arrays."""
    assert SCALAR.check(5.0 * u.s, "time") is None
    assert SCALAR.check([1.0, 2.0] * u.s, "time") is not None


def test_array_shape_accepts_array_rejects_scalar():
    """ARRAY passes arrays and fails scalars."""
    assert ARRAY.check([1.0, 2.0] * u.s, "time") is None
    assert ARRAY.check(5.0 * u.s, "time") is not None


def test_any_shape_accepts_both():
    """ANY_SHAPE imposes no restriction."""
    assert ANY_SHAPE.check(5.0 * u.s, "time") is None
    assert ANY_SHAPE.check([1.0, 2.0] * u.s, "time") is None


def test_shape_singletons_satisfy_shapekind_protocol():
    """The shape singletons are runtime-checkable ShapeKind instances."""
    assert isinstance(SCALAR, ShapeKind)
    assert isinstance(ARRAY, ShapeKind)
    assert isinstance(ANY_SHAPE, ShapeKind)


# ---------------------------------------------------------------------------
# QuantitySpec dataclass
# ---------------------------------------------------------------------------


def test_quantity_spec_defaults():
    """A QuantitySpec has empty equivalencies and constraints by default."""
    spec = QuantitySpec("area", u.m**2)
    assert spec.name == "area"
    assert spec.default == u.m**2
    assert spec.equivalencies == []
    assert spec.constraints == ()


def test_quantity_spec_is_frozen():
    """QuantitySpec is immutable (frozen dataclass)."""
    spec = QuantitySpec("area", u.m**2)
    with pytest.raises(Exception):
        spec.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Spec builder
# ---------------------------------------------------------------------------


def test_spec_build_returns_quantity_spec():
    """Spec.build returns a QuantitySpec with the supplied identity."""
    spec = Spec("gain", u.electron / u.adu).build()
    assert isinstance(spec, QuantitySpec)
    assert spec.name == "gain"
    assert spec.default == u.electron / u.adu
    assert spec.constraints == ()


def test_spec_ranged_adds_range_constraint():
    """Spec.ranged appends a Range constraint."""
    spec = Spec("gain", u.electron / u.adu).ranged(0.1, 100.0).build()
    assert len(spec.constraints) == 1
    assert isinstance(spec.constraints[0], Range)
    assert spec.constraints[0].vmin == 0.1
    assert spec.constraints[0].vmax == 100.0


def test_spec_finite_adds_finite_constraint():
    """Spec.finite appends a Finite constraint."""
    spec = Spec("time", u.s).finite().build()
    assert len(spec.constraints) == 1
    assert isinstance(spec.constraints[0], Finite)


def test_spec_with_constraint_adds_custom_constraint():
    """Spec.with_constraint appends an arbitrary constraint."""
    custom = Range(1.0, 2.0)
    spec = Spec("time", u.s).with_constraint(custom).build()
    assert spec.constraints == (custom,)


def test_spec_chaining_preserves_order():
    """Chained builder calls accumulate constraints in call order."""
    spec = Spec("time", u.s).ranged(0.0, 10.0).finite().build()
    assert len(spec.constraints) == 2
    assert isinstance(spec.constraints[0], Range)
    assert isinstance(spec.constraints[1], Finite)


def test_spec_builder_accepts_equivalencies():
    """Spec forwards equivalencies to the produced QuantitySpec."""
    equivs = u.dimensionless_angles()
    spec = Spec("solid_angle", u.sr, equivs).build()
    assert spec.equivalencies == equivs


# ---------------------------------------------------------------------------
# check_quantity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [10.0 * u.m**2, 0.010 * u.km**2])
def test_check_quantity_valid_returns_equal_value(q):
    """check_quantity returns the same physical quantity for valid input."""
    assert u.isclose(check_quantity(q, AREA), q)


def test_check_quantity_converts_to_canonical_unit():
    """check_quantity converts the result to the spec's canonical unit."""
    result = check_quantity(0.010 * u.km**2, AREA)
    assert result.unit == u.m**2
    assert result.value == pytest.approx(10000.0)


def test_check_quantity_rejects_non_quantity():
    """A non-Quantity value raises TypeError."""
    with pytest.raises(TypeError):
        check_quantity(10.0, AREA)


def test_check_quantity_rejects_non_spec():
    """A non-QuantitySpec spec raises TypeError."""
    with pytest.raises(TypeError):
        check_quantity(10.0 * u.m**2, "not a spec")


def test_check_quantity_rejects_incompatible_unit():
    """An incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        check_quantity(10.0 * u.m, AREA)


def test_check_quantity_honors_equivalencies():
    """A spec's equivalencies enable otherwise-incompatible conversions."""
    result = check_quantity(1.0 * u.rad, SOLID_ANGLE)
    assert result.unit == u.sr


def test_check_quantity_runs_value_constraints():
    """A value violating a constraint raises QuantityValidationError."""
    spec = Spec("gain", u.electron / u.adu).ranged(0.1, 100.0).build()
    check_quantity(1.0 * u.electron / u.adu, spec)
    with pytest.raises(QuantityValidationError):
        check_quantity(1e3 * u.electron / u.adu, spec)


def test_check_quantity_aggregates_multiple_failures():
    """Multiple constraint failures are collected into one error."""
    spec = Spec("time", u.s).ranged(0.0, 10.0).finite().build()
    with pytest.raises(QuantityValidationError) as excinfo:
        check_quantity(np.inf * u.s, spec)
    # inf is both out of range and non-finite -> two problems.
    assert len(excinfo.value.problems) == 2


def test_check_quantity_scalar_shape_enforced():
    """The SCALAR shape rejects arrays and accepts scalars."""
    check_quantity(5.0 * u.s, TIME, SCALAR)
    with pytest.raises(QuantityValidationError):
        check_quantity([1.0, 2.0] * u.s, TIME, SCALAR)


def test_check_quantity_array_shape_enforced():
    """The ARRAY shape rejects scalars and accepts arrays."""
    check_quantity([1.0, 2.0] * u.s, TIME, ARRAY)
    with pytest.raises(QuantityValidationError):
        check_quantity(5.0 * u.s, TIME, ARRAY)


def test_check_quantity_shape_defaults_to_any():
    """Omitting the shape argument allows both scalars and arrays."""
    assert check_quantity(5.0 * u.s, TIME).isscalar
    assert not check_quantity([1.0, 2.0] * u.s, TIME).isscalar


def test_check_quantity_combines_value_and_shape_failures():
    """Value-constraint and shape failures are aggregated together."""
    spec = Spec("time", u.s).ranged(0.0, 10.0).build()
    with pytest.raises(QuantityValidationError) as excinfo:
        check_quantity([1.0, 20.0] * u.s, spec, SCALAR)
    assert len(excinfo.value.problems) == 2


# ---------------------------------------------------------------------------
# _spec_from_annotated / _extract_spec
# ---------------------------------------------------------------------------


def test_spec_from_annotated_finds_spec():
    """_spec_from_annotated pulls a QuantitySpec out of Annotated metadata."""
    from typing import Annotated

    annotation = Annotated[u.Quantity, AREA, Scalar]
    assert _spec_from_annotated(annotation) is AREA


def test_spec_from_annotated_returns_none_without_spec():
    """_spec_from_annotated returns None when no QuantitySpec is present."""
    from typing import Annotated

    annotation = Annotated[u.Quantity, "meta"]
    assert _spec_from_annotated(annotation) is None


def test_spec_from_annotated_returns_none_for_plain_type():
    """_spec_from_annotated returns None for a non-Annotated type."""
    assert _spec_from_annotated(u.Quantity) is None


@pytest.mark.parametrize(
    "annotation,expected_spec,expected_shape",
    [
        (Area, AREA, ANY_SHAPE),
        (Area[Scalar], AREA, SCALAR),
        (Area[Array], AREA, ARRAY),
        (Area[AnyShape], AREA, ANY_SHAPE),
        (Time, TIME, ANY_SHAPE),
        (None, None, ANY_SHAPE),
        (Area[Scalar] | None, AREA, SCALAR),
    ],
)
def test_extract_spec(annotation, expected_spec, expected_shape):
    """_extract_spec resolves alias, shape marker, and union forms."""
    spec, shape = _extract_spec(annotation)
    assert spec == expected_spec
    assert shape is expected_shape


def test_extract_spec_direct_annotated():
    """_extract_spec handles a directly written Annotated hint."""
    from typing import Annotated

    spec, shape = _extract_spec(Annotated[u.Quantity, AREA])
    assert spec is AREA
    assert shape is ANY_SHAPE


def test_extract_spec_no_spec_returns_none():
    """_extract_spec returns (None, ANY_SHAPE) for an unrelated type."""
    spec, shape = _extract_spec(int)
    assert spec is None
    assert shape is ANY_SHAPE


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,name,unit",
    [
        (AREA, "area", u.m**2),
        (TIME, "time", u.s),
    ],
)
def test_catalogue_specs(spec, name, unit):
    """Catalogue constants carry the expected name and canonical unit."""
    assert spec.name == name
    assert spec.default == unit
