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
unit-enforced state (`height`, `zenith_angle`, `rotation_angle`,
`pointing_angle`). Derived read-only geometry properties
(`nadir_angle`, `distance`, `orbital_velocity`,
`orbital_angular_velocity`, `perpendicular_velocity`,
`perpendicular_angular_velocity`, `solid_angle`) are computed from
this state. `calculate_pixel_time(pixel_scale)` converts
`perpendicular_angular_velocity` to pixel traversal time.
`get_tracked_profile(psf, pupil)` convolves the object's own profile
with the pupil defocus profile and a galsim PSF. `_project` applies
line-of-sight foreshortening (scaled by
`mu = cos(nadir_angle - pointing_angle)`) and rotation, and is applied
unconditionally to every object's profile. `CircularOrbitalObject`
builds its profile from a `galsim.TopHat`; `RectangularOrbitalObject`
from a `galsim.Box`; both are returned as the projected
`galsim.Transformation`.

**Pointing angle is a continuum, not a mode.** A satellite's orientation
toward the observer is a continuous tilt described by `pointing_angle`,
measured from the object's nadir direction toward the telescope line of
sight. `pointing_angle = 0` (the default) is nadir-pointing;
`pointing_angle = nadir_angle` is "observatory-pointing" (face-on to the
telescope). The physically meaningful range is `[0, nadir_angle]`, and
`nadir_angle` itself depends on the object's `zenith_angle` and `height`.
The setter validates this range and raises `ValueError` outside it.

## Design Decisions

`orbital_velocity` assumes a circular orbit at `height`:
`sqrt(G * M_earth / (R_earth + height))`. Orbital eccentricity and
inclination are not modelled.

**Projection is always applied.** `_project` runs for every profile;
there is no longer a boolean toggle. The foreshortening factor
`mu = cos(nadir_angle - pointing_angle)` equals `cos(nadir_angle)` at
the nadir default (`pointing_angle = 0`) and `1` (an identity-scale
transform) at the observatory extreme (`pointing_angle = nadir_angle`),
so the single continuous expression strictly generalizes the two former
branches. Historically orientation was a boolean `nadir_pointing` with
`_project` skipped entirely for observatory pointing; the continuous
angle subsumes both former states.

## Invariants

The orbital-mechanics derivations (nadir angle, distance,
perpendicular velocity) are mirrored in
`tests/profiles/test_orbital_objects.py`. That test file is the
authoritative spec for the geometry; keep it in sync with any changes
to the derivations in `orbital_objects.py`.

`mu <= 1` must hold. The `/ mu` term in `_project` conserves total flux
under foreshortening; `mu > 1` would spuriously amplify flux. This is
why `pointing_angle` is constrained to `[0, nadir_angle]`: outside that
range `mu` would exceed `1`. At `zenith_angle = 0` the object is
directly overhead, `nadir_angle` collapses to `0`, and the only valid
`pointing_angle` is `0` (`mu = 1`).

`RectangularOrbitalObject.__init__` is not decorated with
`@enforce_units`, unlike `CircularOrbitalObject.__init__`. This means
`width` and `length` are only validated lazily when their property
getters run — a bad-unit value can be stored and raise later
(tracked in issue #13).
