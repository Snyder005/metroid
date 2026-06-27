# `metroid.profiles`

Geometry and surface-brightness profiles for telescope pupils and orbiting
objects. This is where the orbital mechanics and the `galsim` profile
construction live.

## Modules

### `pupils.py` — telescope apertures

- **`Pupil`** (ABC). Holds a class-level `_registry` of subclasses keyed by a
  `pupil_type` string (registered via `__init_subclass__(pupil_type=...)`).
  `Pupil.from_config(config)` reads `config["type"]`, looks up the subclass, and
  delegates to that subclass's `_from_config`. Abstract members: `area`,
  `get_profile(distance)`, `_from_config`.
- **`CircularPupil`** (`pupil_type="circular"`) — single `radius`; profile is a
  `galsim.TopHat`.
- **`AnnularPupil`** (`pupil_type="annular"`) — `inner_radius` / `outer_radius`
  (validates `outer > inner`); profile is a difference of two `TopHat`s
  (`galsim.Sum`), with the inner disk flux-weighted by `(r_i / r_o)**2` to
  produce a flat annulus.
- `get_profile(distance)` converts a physical aperture size to an angular size
  via `(radius / distance).to_value(u.arcsec, equivalencies=dimensionless_angles())`.

### `orbital_objects.py` — orbiting objects

- **`OrbitalObject`** (ABC). State: `height`, `zenith_angle`, `rotation_angle`,
  `nadir_pointing` (all mutable via unit-enforced setters; `nadir_pointing` is a
  plain bool). Derived read-only geometry:
  - `nadir_angle` — `arcsin(R_earth sin(zenith) / (R_earth + height))`.
  - `distance` — telescope-to-object range (special-cased to `height` when
    `zenith_angle ≈ 0`).
  - `orbital_velocity` — `sqrt(G M_earth / (R_earth + height))` (assumes a
    circular orbit).
  - `orbital_angular_velocity`, `perpendicular_velocity`,
    `perpendicular_angular_velocity`, `solid_angle`.
  - `calculate_pixel_time(pixel_scale)` — time to cross one pixel.
  - `get_tracked_profile(psf, pupil)` — convolves the object profile with the
    pupil defocus profile and the PSF (`galsim.Convolve`).
  - `_project(profile)` — applies foreshortening (cos(nadir_angle)) and rotation
    when `nadir_pointing` is set.
- **`CircularOrbitalObject`** — `radius`; profile `galsim.TopHat`.
- **`RectangularOrbitalObject`** — `width` / `length`; profile `galsim.Box`.

## Dependencies

- External: `galsim`, `astropy.units`, `astropy.constants` (`G`, `R_earth`,
  `M_earth`).
- Internal: `metroid.utils` (`enforce_units`, quantity aliases, `get_field_value`).

## Known issues / gotchas

- **`__init__.py` `__all__` typo.** `__all__` listed `"RectangularOrbit"`, which
  is not a real symbol (the class is `RectangularOrbitalObject`), so
  `from metroid.profiles import *` raised. Fixed on branch
  `documentation/dev-context-and-cleanup`; verify before relying on star
  imports.
- **`RectangularOrbitalObject.__init__` is not decorated with `@enforce_units`**,
  unlike `CircularOrbitalObject.__init__`. Its `width`/`length` are only
  validated lazily when the property getters run, so a bad-unit value can be
  stored and only fail later. Consider adding the decorator for consistency.
- `orbital_velocity` assumes a circular orbit at `height`; there is no
  eccentricity / inclination model yet.
- The orbital-mechanics derivations (nadir angle, distance, perpendicular
  velocity) are mirrored in `tests/profiles/test_orbital_objects.py` — that test
  is the spec for the geometry; keep them in sync.
