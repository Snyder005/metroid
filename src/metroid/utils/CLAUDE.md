# `metroid.utils`

The unit-enforcement machinery that the rest of the package is built on. If you
touch validation or add a new physical quantity, it happens here.

## Modules

### `quantities.py` — the quantity system

The spec is a small **constraint algebra**: a physical identity plus a list of
pluggable checks. Scalar-vs-array is a *separate* axis (a computer
representation, not physics) supplied at the annotation site, not stored in the
spec.

- **`QuantitySpec`** — a frozen dataclass describing one physical quantity:
  `name`, `default` unit, `equivalencies`, and an ordered
  `constraints: tuple[Constraint, ...]`. No range/shape fields.
- **`Constraint` protocol** — any object with
  `check(quantity, name) -> str | None` (message or `None`). Built-ins:
  `Range(vmin, vmax)`, `Finite`. Add a new kind by writing a ~4-line frozen
  dataclass; `check_quantity` never changes.
- **`Spec` builder** — fluent, terse: `Spec("gain", u.electron/u.adu).ranged(0.1, 100).build()`.
- **Shape axis** — `ShapeKind` protocol with singletons `SCALAR` / `ARRAY` /
  `ANY_SHAPE`, and marker *types* `Scalar` / `Array` / `AnyShape` used as the
  generic subscript (`Time[Scalar]`).
- **`check_quantity(quantity, spec, shape=ANY_SHAPE)`** — the validator.
  Raises `TypeError` (not a `Quantity`/`spec`), `ValueError` (incompatible
  unit). Converts to the default unit, then runs all value-level constraints
  **and** the shape check, aggregating every failure into one
  `QuantityValidationError` (a `ValueError` subclass — existing
  `except ValueError` still catches it).
- **`_extract_spec(annotation)`** — returns a `(spec, shape)` tuple. Handles a
  bare generic alias (`Time` → any shape), a subscripted alias
  (`Time[Scalar]`), a raw `Annotated`, and unions (`Time[Scalar] | None`,
  both `typing.Union` and PEP 604 `|`). `(None, ANY_SHAPE)` when no spec.
- The module then defines the canonical `QuantitySpec` constants
  (`WAVELENGTH`, `AREA`, `GAIN`, `PHOTON_FLUX`, ...) and matching **generic**
  type aliases via the `type` statement
  (`type Time[Sh] = Annotated[u.Quantity, TIME, Sh]`, etc.). **These aliases
  are the public vocabulary** used in every annotated signature across `src/`.
  Use the bare alias (`Area`) for any-shape, or `Area[Scalar]` / `Area[Array]`
  to restrict.

### `decorators.py`

- **`enforce_units(func)`** — wraps a callable; binds args, applies defaults,
  and runs `check_quantity` on every parameter whose hint carries a spec, plus
  the return value. Uses the `(spec, shape)` pair from `_extract_spec`, so a
  `Time[Scalar]` hint enforces shape too. Skips `self` and `None` values. Works
  on functions, methods, and properties (stack `@property` above
  `@enforce_units`).
- **`validated_dataclass(**dc_kwargs)`** — a `dataclass` wrapper that also runs
  `enforce_units` on the generated `__init__`. Used by
  `photometry.PhotometricParameters`.

### `validation.py`

- **`get_field_value(config, name, dtype)`** — typed accessor for config dicts.
  Raises `ValueError` for a missing key, `TypeError` for a wrong value/`name`
  type. Used by the `Pupil._from_config` implementations.

## Adding a new physical quantity (the intended workflow)

1. Add a `QuantitySpec` constant in `quantities.py` with the `Spec` builder
   (name, default unit, any equivalencies, any value-level constraints such as
   `.ranged(...)` / `.finite()`).
2. Add the matching generic alias:
   `type NewQuantity[Sh] = Annotated[u.Quantity, NEW_QUANTITY, Sh]`.
3. Annotate parameters/returns with the alias (`NewQuantity` for any shape, or
   `NewQuantity[Scalar]` / `NewQuantity[Array]`) and decorate with
   `@enforce_units`. No bespoke unit checking elsewhere.

To add a new *kind of value-level check*, write a small frozen dataclass with a
`check(quantity, name) -> str | None` method and attach it via the builder's
`.with_constraint(...)`; `check_quantity` needs no changes.

## Known issues / gotchas

- **`validation.py` f-string lint.** Line ~30 used `f"name must be 'str'"` with
  no placeholder (flake8 F541) and the missing-field message lacked the quoting
  the other messages use. Cleaned up on branch
  `documentation/dev-context-and-cleanup`.
- **Range limits were dropped during the constraint-algebra refactor.** No
  catalogue spec currently carries a `Range`; ranges (including the old
  `QUANTUM_EFFICIENCY` `(1e0, 1e2)` that rejected a physically reasonable
  `qe < 1.0`) will be reworked per physical quantity on a case-by-case basis.
  Re-add them with `Spec(...).ranged(vmin, vmax)`.
- **`PHOTON_FLUX` carries a custom `(u.ph, None)` equivalency** so photons are
  treated as dimensionless-countable. Anything comparing/parsing photon-flux
  quantities must pass these equivalencies or conversions will fail.
- **Shape markers come in two flavours**, easy to mix up: the *types* `Scalar`
  / `Array` / `AnyShape` are only ever used as the generic subscript
  (`Time[Scalar]`); the *singletons* `SCALAR` / `ARRAY` / `ANY_SHAPE` are the
  `ShapeKind` instances passed to `check_quantity` directly. `_extract_spec`
  maps the former to the latter.
- **Generic aliases require Python 3.12+** (`type` statement / PEP 695). The
  repo runs 3.13, and `enforce_units` / `validated_dataclass` already use PEP
  695 generics, so this is consistent with the codebase.
- `utils/__init__.py` is empty — import from the submodules by full path
  (`from metroid.utils.quantities import ...`).
