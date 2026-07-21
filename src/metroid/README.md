# metroid

## Overview

`metroid` simulates streaks and trails in astronomical images caused by
orbital objects such as satellites and space debris. It models the
geometry of the object and telescope, computes surface-brightness
profiles via galsim, and scales them to detector ADU through a
radiometry layer.

## Architecture

The simulation pipeline has two parallel threads that combine at the
end:

```
OrbitalObject (geometry, velocity, surface-brightness profile)
        +  Pupil (telescope aperture -> defocus profile)
        +  PSF (galsim)                       --> tracked galsim profile
ThroughputCurve + Sed + PhotometricParameters --> flux / ADU scaling
```

`Observatory` (`observatory.py`) composes a `Camera`, a `Pupil`, and
an `astropy.coordinates.EarthLocation`. Its `get_photo_params(exptime)`
method builds a `PhotometricParameters` instance from camera gain and
qe combined with pupil area — the single entry point for photometric
scaling.

`Camera` (`camera.py`) is an immutable container of named bandpasses
(`dict[str, ThroughputCurve]` wrapped in `MappingProxyType`) plus
gain, `pixel_scale`, and qe. It supports item access by filter name
(`camera[name]`), iteration over filter names, and `len()`.

Subpackages:

- `profiles/` — pupil geometry and orbital object surface-brightness
  profiles
- `photometry/` — flux and ADU calculation
- `utils/` — unit enforcement machinery shared across the package

## Design Decisions

**Unit enforcement is the central pattern.** Almost every public
method and property is decorated with `@enforce_units` (from
`utils/decorators.py`), reading quantity type aliases from
`utils/quantities.py`. To add a new physical quantity, define a
`QuantitySpec` constant and a matching generic alias in
`utils/quantities.py` rather than writing bespoke unit checks inline.

**Immutability is intentional.** Read-only properties, `MappingProxyType`
for the bandpass dict, frozen dataclasses, and frozen numpy arrays are
all deliberate choices. Preserve this pattern when adding new classes.

## Invariants

`Observatory.__init__` validates its arguments with `isinstance` and
raises `ValueError` (not `TypeError`) on a bad type — this is
inconsistent with the rest of the codebase and is currently relied on
by tests (tracked in issue #23).

`Camera.__getitem__` raises `ValueError`, not `KeyError`, for an
unknown bandpass name (tracked in issue #24).
