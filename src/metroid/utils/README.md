# utils

## Overview

The unit-enforcement machinery the rest of the package is built on.
It provides a small constraint algebra for physical quantities,
decorator-based enforcement on function parameters and return values,
and a typed config-dict accessor.

## Architecture

A `QuantitySpec` (`quantities.py`) captures the essential identity of
a physical quantity: a name, a canonical `astropy` unit, optional unit
equivalencies, and an ordered tuple of pluggable `Constraint` checks.
The `Spec` fluent builder keeps catalogue declarations terse (e.g.
`Spec("gain", u.electron / u.adu).ranged(0.1, 100).build()`).

Scalar-vs-array is a separate axis from the physical spec. Shape
restrictions are expressed at the annotation site via generic
subscripts (`Time[Scalar]`, `Time[Array]`) and resolved at runtime by
`ShapeKind` singletons (`SCALAR`, `ARRAY`, `ANY_SHAPE`).

`check_quantity` converts the quantity to the spec's canonical unit,
runs every value-level `Constraint`, and runs the shape check. All
failures are collected and raised together as a single
`QuantityValidationError` (a `ValueError` subclass).

`_extract_spec` maps a type hint — bare alias (`Time`), subscripted
alias (`Time[Scalar]`), raw `Annotated`, or union — to a
`(QuantitySpec, ShapeKind)` pair. This is what makes
`@enforce_units` annotation-driven.

The catalogue at the bottom of `quantities.py` defines
`QuantitySpec` constants and matching generic `type` aliases
(`Time`, `Area`, `Gain`, `Wavelength`, etc.). These aliases are the
public vocabulary used across the package.

`enforce_units` (`decorators.py`) binds arguments, applies defaults,
runs `check_quantity` on every annotated parameter, calls the function,
then validates the return value. `validated_dataclass` extends the same
enforcement to a dataclass `__init__`.

`get_field_value` (`validation.py`) is a typed accessor for
configuration dictionaries; it is used by `Pupil._from_config` and
similar config-driven constructors.

## Design Decisions

**Adding a new physical quantity** follows a three-step workflow:
(1) add a `QuantitySpec` constant via the `Spec` builder,
(2) add the matching generic alias
`type NewQuantity[Sh] = Annotated[u.Quantity, NEW_QUANTITY, Sh]`,
(3) annotate parameters and return values and decorate the callable
with `@enforce_units`. No bespoke unit checks are needed elsewhere.

**Adding a new kind of value-level check** requires writing a small
frozen dataclass with `check(quantity, name) -> str | None` and
attaching it via `.with_constraint(...)`. The `check_quantity`
function never changes.

**Range limits are intentionally absent** from the current catalogue.
An earlier `QUANTUM_EFFICIENCY` range rejected physically reasonable
values below 1.0 (see closed issue #14). Ranges will be reworked per
quantity case by case.

## Invariants

Two flavors of shape markers exist and must not be mixed: the marker
types `Scalar`, `Array`, and `AnyShape` are used only as generic
subscripts at annotation sites; the singletons `SCALAR`, `ARRAY`, and
`ANY_SHAPE` are passed as the `shape` argument to `check_quantity`.

`PHOTON_FLUX` carries a custom `(u.ph, None)` equivalency so that
photons are treated as dimensionless-countable. Any code that
compares or converts photon flux quantities must supply these
equivalencies.

Generic type aliases (`type Wavelength[Sh] = ...`) require Python
3.12+ (PEP 695). The repository targets Python 3.13.

`utils/__init__.py` is empty. Import from submodules by their full
path (e.g. `from metroid.utils.quantities import Time`).
