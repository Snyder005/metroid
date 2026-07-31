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

**Composite objects and components.** `CompositeOrbitalObject` models a
real satellite as an assembly of parts. Each part is a `Component`
(`CircularComponent` → `galsim.TopHat`, `RectangularComponent` →
`galsim.Box`) described in the satellite's own *body frame*: a shape, a
local centroid offset `(x0, y0)` in meters, and a `reflectivity`. A
`Component` deliberately holds **no** orbital state — distance, velocity,
and projection all belong to the enclosing `CompositeOrbitalObject`,
which shares one orbit for the whole rigid body. This mirrors how `Pupil`
is orbit-agnostic and receives `distance` at `get_profile` time. The
composite's `profile` builds each component profile at the shared
`distance`, sums them with `galsim.Sum`, and projects the sum once; its
`area` is the sum of component areas.

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

**Reflectivity sets relative surface brightness.** Under a fully diffuse
(Lambertian) assumption a component's reflected surface brightness
(radiance) is proportional to its `reflectivity`, and its total reflected
signal is proportional to `reflectivity * area`. That product is used as
each component's galsim flux (`Component.relative_flux`), so `galsim.Sum`
yields physically correct *relative* brightness between parts (a large
dim panel vs. a small bright bus). This is a *relative* model only;
absolute photometric normalization (magnitude → ADU) is intentionally
deferred to the flux-scaling roadmap (issue #35). The reflectivity → flux
computation is isolated in `relative_flux` so #35 can extend it.

`CompositeOrbitalObject.area` is the simple sum of component areas.
Overlap between parts is ignored (second order under the flat-diffuse
assumption), which keeps `solid_angle = area / distance^2` well defined
at the cost of slight over-counting when parts physically overlap.

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

**Project once, on the sum.** The line-of-sight foreshortening
(`_project`) must be applied exactly once, to the summed composite
profile, never per component. Components return unprojected body-frame
profiles (`Component.get_profile` has no orbital state and cannot
project); only `CompositeOrbitalObject.profile` projects. Violating this
double-projects and corrupts the geometry.
