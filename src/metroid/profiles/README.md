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
unconditionally to every object's profile; the foreshortening is *not*
flux-conserving (total flux dims by `mu`, see Invariants).
`calculate_flux`, `get_scaled_profile`, and `get_scaled_tracked_profile`
give the profile an absolute scale from an AB magnitude (see **Flux
scaling** below). `CircularOrbitalObject`
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
`distance`, projects each one with the shared line-of-sight angle, and
sums them with `galsim.Sum`; its `area` is the sum of component areas.
(Projecting per component before summing is a *seam* for future work — see
Invariants.)

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
dim panel vs. a small bright bus). This is a *relative* model; the
absolute scale comes from **Flux scaling** below.

**Flux scaling — a brightness magnitude is required.** Observatories
report magnitudes, so an object's absolute brightness is set by an AB
magnitude (or an `Sed`), never a raw flux. A magnitude is **required** at
construction, supplied as exactly one of two mutually exclusive keyword
arguments — never both, so a caller cannot supply a pair that violates
their fixed geometric relationship:

- `observed_magnitude` — the magnitude of a specific satellite measured for
  a specific observation, i.e. at the geometry given by the other
  construction parameters.
- `canonical_magnitude` — a standardized "average" magnitude at the
  canonical reference geometry (`CANONICAL_HEIGHT` = 500 km observed at
  zenith). There is no astronomical standard for this reference yet, so the
  height lives in a single module-level global.

Either way the geometry-invariant **canonical** magnitude is what gets
stored (an `observed_magnitude` is converted to canonical at the
construction geometry first), because `height`/`zenith_angle`/`pointing_angle`
are mutable and a stored observed magnitude would go stale. The conversion
is purely geometric (a flux ratio, band-independent):
`m_can = m_obs + 2.5*log10(mu) - 5*log10(distance / d_can)`, with
`flux ∝ mu / distance^2`. `canonical_magnitude` and `observed_magnitude`
(re-derived for the current geometry) are read-only properties.

**Projection lives only in the standardization, not in the collected
flux.** Current studies identify a specific satellite and measure its
magnitude *while tracking* it. That measurement is made knowing only the
zenith angle and height: the observatory does not know the satellite's
pointing/orientation and assumes all reflected flux is collected regardless
of how the projection spread it across the profile. So the
`observed_magnitude → total ADU` path (`calculate_flux`) is **independent of
projection** — `mu` does not appear in it. Projection (`mu`) enters *only*
the `canonical ↔ observed` conversion above. This is deliberate and
essential: standardizing an observed magnitude to a common geometry is
exactly the correction this software exists to compute, since the
"average/canonical" magnitudes reporting observatories publish do not
account for projection (nor, separately, for an angle-dependent BRDF — a
future correction). Within this corrected framework we place satellites
with as much known geometry/orientation/BRDF information as possible to
predict what future observatories will measure.

At render time the observed magnitude routes through the photometry layer
(`ThroughputCurve.calculate_adu`) to a total **ADU**, which becomes the
galsim profile flux (metroid convention: galsim flux ≡ total ADU; valid
because the profile is noiseless — if Poisson noise is ever added it must
be applied in electrons before `/gain → ADU`). `get_scaled_profile` scales
the bare profile; `get_scaled_tracked_profile` scales the *convolved*
profile — the flux must be applied **after** convolution because the pupil
defocus profile is not unit-flux (`AnnularPupil` carries `1 - (r_i/r_o)^2`)
and `withFlux` normalizes to unit flux before rescaling to the target ADU.
For a composite, `withFlux` on the summed profile distributes the total
across components as `total_adu * w_i / sum(w_j)` (GalSim linearity), so no
per-component flux loop is needed. The reflectivity → flux computation
stays isolated in `Component.relative_flux` as a *relative* weight.

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

`0 < mu <= 1` must hold. `pointing_angle` is constrained to
`[0, nadir_angle]` so that `mu = cos(nadir_angle - pointing_angle)` stays
in `(0, 1]`. At `zenith_angle = 0` the object is directly overhead,
`nadir_angle` collapses to `0`, and the only valid `pointing_angle` is `0`
(`mu = 1`).

**Projection is not flux-conserving.** `_project` foreshortens along one
axis with `transform(mu, 0, 0, 1)`, whose Jacobian scales total flux by
`mu`; there is deliberately no compensating `/ mu`. A diffuse (Lambertian)
surface seen off-normal reflects less total light toward the observer in
proportion to its projected area, so total flux dims by `mu`. (This
relaxes the earlier convention, where `/ mu` conserved flux.) For an
absolutely-scaled profile the dimming is invisible because
`get_scaled_profile`/`get_scaled_tracked_profile` set the final total with
`withFlux`; it is observable only in the bare `profile` flux (a
foreshortened primitive has flux `mu`, not `1`) and in future
per-component projected-area weighting.

`RectangularOrbitalObject.__init__` is not decorated with
`@enforce_units`, unlike `CircularOrbitalObject.__init__`. This means
`width` and `length` are only validated lazily when their property
getters run — a bad-unit value can be stored and raise later
(tracked in issue #13).

**Project each component once, with the shared angle (the seam).**
`CompositeOrbitalObject.profile` projects each component profile with the
composite's single shared line-of-sight angle and then sums, rather than
projecting the summed profile once. With one shared angle the two are
provably identical (both `_project` and `galsim.Sum` are linear about the
body-frame origin), so this changes nothing today. It exists as a *seam*:
a future model can promote the shared `self._project(...)` to a
per-component projection that computes each component's own `mu` from its
own body-frame normal and the shared viewing geometry. **Revisit the seam
when any of:** (1) components gain independent orientation/pointing (their
own normals), (2) a 3-D body is projected to a 2-D observed profile, or
(3) per-component projected-area flux weighting is required (e.g. a
deployed solar panel seen edge-on). Components themselves still return
*unprojected* body-frame profiles (`Component.get_profile` has no orbital
state and cannot project); each component must be projected exactly once
by the composite. Projecting a component twice corrupts the geometry.
