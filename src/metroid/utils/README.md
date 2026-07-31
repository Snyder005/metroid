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

`config.py` is the declarative configuration layer. `load_yaml` parses
a YAML document into a `dict`; `load_standard_catalogue` resolves the
bundled `metroid/data/standard_objects.yaml` via `importlib.resources`.
`Registrable` is a mixin providing a per-base subclass registry plus the
`from_config`/`_from_config` type dispatch shared by `Pupil`, `Camera`,
and `Observatory`.

## Configuration Layer

**Format-isolated, dict-based core.** The only code that knows the
on-disk format is YAML is `load_yaml` (`yaml.safe_load` → `dict`). Every
`from_config` downstream consumes a plain `dict`, so the format can
change (JSON, TOML) by adding a loader without touching any class, and
tests can bypass files by passing dicts directly.

**Shared subclass registry.** A base opts in with
`registry_label="<noun>"`, which gives that base its own fresh
`_registry` dict; concrete subclasses register with `type="<name>"`.
`from_config` pops the `type` field, looks up the subclass, and delegates
to its `_from_config`. The error messages
(`config is missing required field 'type'`,
`unknown <label> type: ...`) are produced in one place, so `Pupil`'s
original public behavior is preserved after the refactor.

**Name/label vs. type — two distinct selectors.** The **name/label**
(e.g. `"rubin"`) picks *which* standard object from the catalogue;
`Observatory.from_standard(name)` resolves it against the bundled file.
The **type** (e.g. `"annular"`) picks *which subclass* for a polymorphic
object; `from_config` resolves it against the class registry.

**Extension point.** `Registrable` is class-agnostic. `OrbitalObject`
subclasses (including composite satellites) can opt in by passing
`registry_label`/`type` and implementing `_from_config`, enabling
declarative satellite definitions with no new machinery. That join is
deferred to keep scope tight.

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

Config fields carry **bare numbers in each field's documented canonical
unit** (radius in meters, gain in electrons/ADU, lat/lon in degrees);
there is no `{value, unit}` wrapper. Values are converted on load and
then validated by `@enforce_units`.

`Registrable.from_config` copies its input before popping `type`, so a
caller's dict is never mutated. Each hierarchy root (a class with
`registry_label`) owns a separate `_registry`, so sibling hierarchies
never collide.

The bundled `standard_objects.yaml` must ship as package data (declared
under `[tool.setuptools.package-data]` for `metroid.data` in
`pyproject.toml`) or `from_standard` breaks after `pip install`.
`metroid/data/` is a package (has `__init__.py`), so
`importlib.resources.files("metroid.data")` resolves it. `pyyaml` is a
runtime dependency; `import yaml` is isolated to `config.py`.
