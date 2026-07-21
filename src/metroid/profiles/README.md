# profiles

## Overview

Geometry and surface-brightness profiles for telescope pupils and
orbiting objects. This is where orbital mechanics and galsim profile
construction live.

## Architecture

**Pupil hierarchy.** `Pupil` is an ABC with a class-level `_registry`
dict. Concrete subclasses register themselves by passing
`pupil_type="<name>"` to their class statement, which triggers
`__init_subclass__`. `Pupil.from_config(config)` reads `config["type"]`,
looks up the subclass in `_registry`, and delegates construction to
its `_from_config` classmethod. `CircularPupil` produces a
`galsim.TopHat` profile; `AnnularPupil` produces the difference of
two `TopHat` objects (the inner disk's flux is weighted by
`(r_i / r_o)**2` to represent the blocked area). Both convert
physical aperture radius to an angular size using
`dimensionless_angles()` equivalencies at the observed distance.

**OrbitalObject hierarchy.** `OrbitalObject` is an ABC with mutable,
unit-enforced state (`height`, `zenith_angle`, `rotation_angle`) and
a bool flag `nadir_pointing`. Derived read-only geometry properties
(`nadir_angle`, `distance`, `orbital_velocity`,
`orbital_angular_velocity`, `perpendicular_velocity`,
`perpendicular_angular_velocity`, `solid_angle`) are computed from
this state. `calculate_pixel_time(pixel_scale)` converts
`perpendicular_angular_velocity` to pixel traversal time.
`get_tracked_profile(psf, pupil)` convolves the object's own profile
with the pupil defocus profile and a galsim PSF. `_project` applies
foreshortening (scaled by `cos(nadir_angle)`) and rotation when
`nadir_pointing` is `True`. `CircularOrbitalObject` produces a
`galsim.TopHat` profile; `RectangularOrbitalObject` produces a
`galsim.Box`.

## Design Decisions

`orbital_velocity` assumes a circular orbit at `height`:
`sqrt(G * M_earth / (R_earth + height))`. Orbital eccentricity and
inclination are not modelled.

## Invariants

The orbital-mechanics derivations (nadir angle, distance,
perpendicular velocity) are mirrored in
`tests/profiles/test_orbital_objects.py`. That test file is the
authoritative spec for the geometry; keep it in sync with any changes
to the derivations in `orbital_objects.py`.

`RectangularOrbitalObject.__init__` is not decorated with
`@enforce_units`, unlike `CircularOrbitalObject.__init__`. This means
`width` and `length` are only validated lazily when their property
getters run — a bad-unit value can be stored and raise later
(tracked in issue #13).
