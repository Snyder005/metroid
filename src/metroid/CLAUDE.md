# `metroid` — top-level package

Core entry points that compose the rest of the library. See the root
`CLAUDE.md` for project-wide development rules; this file is context for
working inside `src/metroid/` and its subpackages.

## Mental model

`metroid` simulates streaks/trails left by orbiting objects (satellites,
debris) in astronomical images. The pipeline conceptually is:

```
OrbitalObject (geometry, velocity, surface-brightness profile)
        +  Pupil (telescope aperture -> defocus profile)
        +  PSF (galsim)                       --> tracked galsim profile
ThroughputCurve + Sed + PhotometricParameters --> flux / ADU scaling
```

The top-level module wires the optics together:

- **`Observatory`** (`observatory.py`) — composes a `Camera`, a `Pupil`, and an
  `astropy.coordinates.EarthLocation`. Validates each argument by `isinstance`
  in `__init__` (raises `ValueError` on mismatch). `get_photo_params(exptime)`
  builds a `PhotometricParameters` from the camera gain/qe and pupil area.
- **`Camera`** (`camera.py`) — an immutable container of named bandpasses
  (`dict[str, ThroughputCurve]`, wrapped in a `MappingProxyType`) plus `gain`,
  `pixel_scale`, and `qe`. Supports `camera[name]`, iteration, and `len()`.
  Unknown keys raise `ValueError` (not `KeyError`).

## Conventions used everywhere in `src/`

- **Unit enforcement is the central pattern.** Almost every public method and
  property is decorated with `@enforce_units` (from `utils/decorators.py`),
  which reads `Annotated[u.Quantity, QuantitySpec]` hints (the aliases in
  `utils/quantities.py`, e.g. `Gain`, `Time`, `Area`) and validates the
  unit/equivalency/range of inputs *and* the return value. To add a new
  physical quantity, add a `QuantitySpec` + `Annotated` alias in
  `utils/quantities.py` and annotate with it — do not hand-roll unit checks.
- **Immutability is intentional.** Containers expose read-only properties and
  use `MappingProxyType` / frozen dataclasses / frozen numpy arrays. Recent
  history ("more protected immutability") shows this is a deliberate design
  value — preserve it.
- Type hints are required on all code; line length 110 (code) / 79
  (docstrings); numpydoc style.

## Known issues / gotchas (top-level)

- `Observatory.__init__` validates by `isinstance` and raises `ValueError` for
  a bad type, whereas most of the codebase raises `TypeError` for type
  mismatches — inconsistent but currently relied on by tests.
- `Camera.__getitem__` translates a missing key into `ValueError`, so callers
  catch `ValueError`, not `KeyError`.
