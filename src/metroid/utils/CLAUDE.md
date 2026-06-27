# `metroid.utils`

The unit-enforcement machinery that the rest of the package is built on. If you
touch validation or add a new physical quantity, it happens here.

## Modules

### `quantities.py` — the quantity system

- **`QuantitySpec`** — describes one physical quantity: `name`, `default` unit,
  optional `equivalencies`, optional `typical_range` (inclusive min/max on the
  value in default units).
- **`check_quantity(quantity, spec)`** — the validator. Raises `TypeError` if
  not a `Quantity` (or `spec` is wrong type), `ValueError` for an
  incompatible unit or an out-of-range value. Returns the quantity *converted to
  the spec's default unit*. Range checks handle both scalars and arrays.
- **`_extract_spec(annotation)`** — pulls a `QuantitySpec` out of an
  `Annotated[...]` hint, recursing through `Annotated` and `Union`
  (e.g. `Area | None`). Returns `None` when there is no spec.
- The module then defines the canonical `QuantitySpec` constants
  (`WAVELENGTH`, `AREA`, `GAIN`, `PHOTON_FLUX`, ...) and matching
  `Annotated[u.Quantity, SPEC]` type aliases (`Wavelength`, `Area`, `Gain`,
  `PhotonFlux`, ...). **These aliases are the public vocabulary** used in every
  annotated signature across `src/`.

### `decorators.py`

- **`enforce_units(func)`** — wraps a callable; binds args, applies defaults,
  and runs `check_quantity` on every parameter whose hint carries a spec, plus
  the return value. Skips `self` and `None` values. Works on functions,
  methods, and properties (stack `@property` above `@enforce_units`).
- **`validated_dataclass(**dc_kwargs)`** — a `dataclass` wrapper that also runs
  `enforce_units` on the generated `__init__`. Used by
  `photometry.PhotometricParameters`.

### `validation.py`

- **`get_field_value(config, name, dtype)`** — typed accessor for config dicts.
  Raises `ValueError` for a missing key, `TypeError` for a wrong value/`name`
  type. Used by the `Pupil._from_config` implementations.

## Adding a new physical quantity (the intended workflow)

1. Add a `QuantitySpec` constant in `quantities.py` (name, default unit, any
   equivalencies/range).
2. Add the matching `Annotated[u.Quantity, SPEC]` alias.
3. Annotate parameters/returns with the alias and decorate with
   `@enforce_units`. No bespoke unit checking elsewhere.

## Known issues / gotchas

- **`validation.py` f-string lint.** Line ~30 used `f"name must be 'str'"` with
  no placeholder (flake8 F541) and the missing-field message lacked the quoting
  the other messages use. Cleaned up on branch
  `documentation/dev-context-and-cleanup`.
- **`QUANTUM_EFFICIENCY` range is `(1e0, 1e2)`** while its default in
  `Camera`/`PhotometricParameters` is exactly `1.0 electron/ph` — i.e. it sits
  on the lower boundary. A qe below 1.0 (a physically reasonable detector value)
  would be rejected. Confirm this range is intended.
- **`PHOTON_FLUX` carries a custom `(u.ph, None)` equivalency** so photons are
  treated as dimensionless-countable. Anything comparing/parsing photon-flux
  quantities must pass these equivalencies or conversions will fail.
- `check_quantity`'s scalar branch uses `np.isscalar`, which reports `False` for
  0-d quantities. The `validation_frameworks/` directory (untracked, top-level)
  contains three *proposed* rewrites that add explicit scalar-vs-array
  validation; none is integrated into `src/` yet.
- `utils/__init__.py` is empty — import from the submodules by full path
  (`from metroid.utils.quantities import ...`).
