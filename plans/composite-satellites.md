# Composite Satellites Built from Components

## Overview

`metroid` currently models a satellite as a single geometric primitive: `CircularOrbitalObject`
(a `galsim.TopHat`) or `RectangularOrbitalObject` (a `galsim.Box`). Real satellites are assemblies
of distinct parts — a central bus, one or more solar panels, a dish — each with its own shape,
position, and reflectivity. This plan introduces a `Component` abstraction and a
`CompositeOrbitalObject` that assembles a set of positioned, reflectivity-weighted component
profiles into a single `galsim.Sum`, which then flows unchanged through the existing
`get_tracked_profile` convolution/defocus machinery.

The chosen approach keeps GalSim as the sole engine for spatial manipulation: each `Component`
converts its physical dimensions to an angular `galsim.GSObject`, applies a positional `shift`
(from a physical offset within the satellite body frame), and carries a relative flux derived from
a physical **reflectivity × area** surface-brightness model. `CompositeOrbitalObject.profile` sums
these and applies the existing `_project()` transform exactly as the primitive objects do, so the
composite is a drop-in `OrbitalObject`. Per the direction on this task, the plan defines an
absolute-ish physical surface-brightness model now (reflectivity sets radiance) rather than a purely
relative weighting; this deliberately overlaps with the proposed flux-scaling workflow in issue #35
and that overlap is called out in the Decision Log and Known Risks so integration can reconcile the
two without rework surprises.

## Planning Context

This section is consumed VERBATIM by downstream agents (Technical Writer, Quality Reviewer).

### Decision Log

| Decision | Reasoning Chain |
| -------- | --------------- |
| Introduce a `Component` ABC parallel to `OrbitalObject`, not reuse `OrbitalObject` subclasses as components | Components live in the satellite *body frame* and have a local centroid offset + reflectivity, but no independent orbit (height/zenith are shared by the whole satellite) -> reusing `OrbitalObject` would force every component to carry orbital state that is meaningless per-part and duplicated across parts -> a dedicated lightweight `Component` type with only shape + offset + reflectivity is the minimal correct model. |
| Component shapes mirror the existing primitives (`CircularComponent` -> `galsim.TopHat`, `RectangularComponent` -> `galsim.Box`) | The primitive `OrbitalObject` subclasses already encode the exact physical-length-to-angular-size conversion (`radius/distance` -> arcsec via `dimensionless_angles`) -> duplicating that logic in components keeps a single conversion convention across the package -> mirroring the two existing shapes covers the leosim `Panel`/`Bus` (box) and `Dish` (tophat) cases without inventing new geometry. |
| Angular conversion uses the *composite's* `distance`, passed into `Component.get_profile(distance)` | Distance is an orbital property of the whole satellite, not of a component -> passing it in at profile-build time (mirroring `Pupil.get_profile(distance)`) keeps components stateless w.r.t. orbit and avoids storing a back-reference to the parent -> matches the established `Pupil.get_profile` signature so the pattern is already familiar in the codebase. |
| Relative surface brightness is `reflectivity` (dimensionless [0,1]); per-component GalSim flux = reflectivity × physical area (in canonical units) | Reflectivity physically sets radiance (surface brightness) for a diffuse (Lambertian) reflector -> total reflected signal from a flat part scales as radiance × projected area, so flux ∝ reflectivity × area -> encoding flux this way makes `galsim.Sum` produce physically correct *relative* brightness between parts (a large dim panel vs. a small bright bus) without any post-hoc reweighting. |
| Combine via `galsim.Sum` (equivalently the `+` operator on shifted, flux-scaled profiles) | GalSim `Sum` is the canonical additive combination of surface-brightness profiles and preserves each summand's flux -> the summed object is itself a `galsim.GSObject`, so it satisfies the `OrbitalObject.profile` return contract and feeds `galsim.Convolve` in `get_tracked_profile` unchanged -> no changes needed downstream of `profile`. |
| `CompositeOrbitalObject` subclasses `OrbitalObject` and stores a non-empty tuple of `Component`s | Composite must expose the same orbital properties (`distance`, `solid_angle`, velocities) and the same `profile`/`area`/`get_tracked_profile` surface as primitives -> subclassing inherits all orbital mechanics for free and only requires overriding `profile` and `area` -> a tuple (immutable) prevents post-construction mutation that would silently desync cached-looking derived state. |
| `CompositeOrbitalObject.area` = sum of component areas | `area` feeds `solid_angle = area / distance^2`, used by radiance/flux conversions -> the physically meaningful total emitting area of a composite is the sum of its parts' areas (overlap is a second-order effect ignored here, consistent with the diffuse-flat assumption) -> summing is the simplest correct-to-first-order definition and keeps `solid_angle` well defined. |
| Component centroid offset stored as a 2-D physical length `(x0, y0)` in meters, converted to an angular `shift` at profile-build time | leosim stores centroids in meters and shifts in the image plane -> keeping offsets physical (body-frame meters) makes a satellite definition independent of its distance/orientation, so the same satellite can be observed at any geometry -> conversion to arcsec uses the same `offset/distance` `dimensionless_angles` idiom as the size conversion, keeping one convention. |
| `_project()` is applied to the *summed* composite profile, not per component | Projection is a foreshortening of the whole rigid body along the line of sight, governed by the shared `nadir_angle`/`rotation_angle` -> applying it once to the sum is both physically correct (rigid body) and cheaper than per-component -> matches how primitives apply `_project` to their single profile. |

### Rejected Alternatives

| Alternative | Why Rejected |
| ----------- | ------------ |
| Reuse `CircularOrbitalObject`/`RectangularOrbitalObject` instances as components | They carry full orbital state (height, zenith, nadir_pointing) that is nonsensical and duplicated per-part; their `profile` already applies `_project` individually, which would double-project once the composite projects the sum. |
| Pure relative-weight (unitless) component flux with no area term | Ignores that a physically larger part reflects more total light at equal reflectivity; would require users to hand-tune weights to fake the area dependence, re-deriving physics the model already knows. |
| Store distance on each `Component` | Couples components to a specific observation geometry; a satellite definition should be reusable across zenith angles and heights. Passing distance to `get_profile` keeps components geometry-agnostic. |
| Compute absolute photometric flux (magnitudes/ADU) inside the composite now | That is precisely the post-convolution normalization workflow proposed in issue #35; doing it here would fork that design. This plan stops at *relative* physical brightness (reflectivity × area) and leaves absolute normalization to #35. |
| Add a general N-gon / arbitrary polygon component | No current requirement; the two existing primitives cover bus/panel/dish. Extra geometry is speculative surface area. |

### Constraints & Assumptions

- **Existing patterns to follow**: physical-length→arcsec conversion via
  `(length / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())`
  (see `orbital_objects.py:275`, `pupils.py:186`); `@enforce_units` on all quantity-typed
  parameters and properties; read-only properties backed by `_name` attributes; unit specs and
  shape markers from `metroid.utils.quantities`.
- **GalSim is the manipulation engine**: use `galsim.TopHat`, `galsim.Box`, `GSObject.shift`,
  `GSObject.withFlux`/flux kwarg, and `galsim.Sum`. Do not implement pixel math by hand.
- **Reflectivity is dimensionless in [0, 1]**: a new `Reflectivity`/`Fraction` quantity spec.
  `FRACTION` already exists in `quantities.py:372` (`u.dimensionless_unscaled`) and can be reused,
  or a dedicated `REFLECTIVITY` spec added for clarity.
- **Diffuse-flat (Lambertian) assumption**: BRDF is treated as fully diffuse and angle-independent,
  matching the stated scope of issue #35. Reflectivity → radiance is a linear scaling only.
- **Dependencies**: `galsim`, `astropy.units`, `numpy` — all already used in `profiles/`.
- **Package layout**: new code lives in `src/metroid/profiles/`; tests mirror under
  `tests/profiles/`. New public names exported from `src/metroid/profiles/__init__.py`.

### Known Risks

| Risk | Mitigation | Anchor |
| ---- | ---------- | ------ |
| Overlap with issue #35 (flux scaling): the reflectivity×area flux model may be superseded or need reconciliation with the canonical-magnitude workflow. | Accepted and intentional per task direction. Scope this plan to *relative* physical brightness only; keep the reflectivity→flux mapping in one small method so #35 can wrap or replace it. Flag in Invisible Knowledge. | `profiles/orbital_objects.py:271-282` (primitives return a bare, unnormalized `profile`; composite matches this contract). |
| Double projection if a component's `get_profile` applied `_project` itself. | Components must return an *unprojected* body-frame profile; only `CompositeOrbitalObject.profile` calls `_project` on the sum. Enforced by design: `Component` has no orbital state and cannot project. | `profiles/orbital_objects.py:278-282` (primitive applies `_project` exactly once, at the object level). |
| `galsim.Sum` of profiles with differing implicit flux normalization could yield unintended relative brightness if a component's base profile is not unit-flux before scaling. | Explicitly set each component's flux via the reflectivity×area value (e.g. `withFlux`) rather than relying on GalSim defaults; document that `galsim.TopHat`/`Box` default to unit total flux. | — (external GalSim behavior; verify in tests). |
| Shift direction/sign convention (body-frame `(x0, y0)` vs. GalSim image axes, and interaction with `rotation_angle`) could place components incorrectly. | Define and test the convention explicitly: offsets are in the *unrotated body frame*; `_project` (which rotates by `rotation_angle`) is applied to the already-assembled sum so components rotate rigidly with the body. Add a test asserting relative centroid positions. | `profiles/orbital_objects.py:241-244` (`_project` rotates by `phi`, transforms, rotates back). |
| `area` as a simple sum ignores component overlap, over-counting emitting area and inflating `solid_angle`. | Accepted to first order under the flat-diffuse assumption; documented. Overlap correction is out of scope. | — |

## Invisible Knowledge

Technical Writer: create/update `src/metroid/profiles/README.md`.

1. **Architectural decision — body frame vs. orbit frame**: A `Component` describes a part of a
   satellite in the satellite's own *body frame* (a local centroid offset in meters and a shape in
   meters). It deliberately has **no** orbital state. All orbital mechanics (distance, velocity,
   projection) belong to the enclosing `CompositeOrbitalObject`, which shares one height/zenith for
   the whole rigid body. This mirrors how `Pupil` is orbit-agnostic and receives `distance` at
   `get_profile` time.

2. **Business rule — reflectivity sets relative surface brightness**: Under a fully diffuse
   (Lambertian) assumption, a component's reflected surface brightness (radiance) is proportional
   to its reflectivity, and its total reflected signal is proportional to `reflectivity × area`.
   That product is the per-component GalSim flux, so `galsim.Sum` yields physically correct
   *relative* brightness between parts. This is a *relative* model only — absolute photometric
   normalization (magnitude → ADU) is intentionally deferred to the flux-scaling roadmap (issue
   #35). Keep the reflectivity→flux computation isolated so #35 can extend it.

3. **System invariant — project once, on the sum**: The line-of-sight foreshortening `_project`
   must be applied exactly once, to the summed composite profile, never per component. Components
   return unprojected body-frame profiles. Violating this double-projects and corrupts geometry.

4. **Invariant — components combine additively and non-destructively**: `galsim.Sum` preserves each
   summand's flux; the composite `profile` is a plain `galsim.GSObject` and therefore a drop-in for
   the primitive `.profile`, flowing through `get_tracked_profile`'s `galsim.Convolve` with no
   downstream changes.

5. **Tradeoff — `area` is the sum of part areas**: Overlap between components is ignored (second
   order under flat-diffuse). This keeps `solid_angle = area / distance^2` well defined and simple
   at the cost of slight over-counting when parts physically overlap.

## Milestones

### Milestone 1: Component abstraction

**Files**: `src/metroid/profiles/components.py` (new),
`src/metroid/utils/quantities.py` (optional new `REFLECTIVITY` spec)

**Code Intent**:

- Add a `Component` ABC describing one part of a satellite in the body frame. Constructor
  (unit-enforced) takes: a 2-D centroid offset — model as two scalar `GeometryLength[Scalar]`
  parameters `x0`, `y0` (default `0 m`) — and a `reflectivity` (dimensionless in `[0, 1]`; reuse
  `Fraction`/`FRACTION` or add a dedicated `REFLECTIVITY = Spec("reflectivity", u.dimensionless_unscaled).ranged(0, 1)`).
- Read-only, unit-enforced properties: `x0`, `y0`, `reflectivity`.
- Abstract read-only property `area -> Area[Scalar]` (physical part area).
- Concrete method `relative_flux() -> float` (or property): returns
  `(self.reflectivity * self.area).to_value(<canonical>)` as the GalSim flux weight. Keep this in
  one small method (Decision Log: reflectivity→flux; supports issue #35 extension).
- Abstract method `get_profile(distance: OrbitalDistance[Scalar]) -> galsim.GSObject` returning the
  **unprojected** body-frame profile: build the angular shape, set flux to `relative_flux()`
  (e.g. `profile.withFlux(self.relative_flux())`), then `shift` by the angular offset
  `(x0/distance, y0/distance)` converted to arcsec via `dimensionless_angles`.
- Concrete subclasses:
  - `CircularComponent(radius, x0=0, y0=0, reflectivity=1)` → `galsim.TopHat(r)`,
    `area = pi r^2` (mirror `CircularOrbitalObject`, `orbital_objects.py:270-290`).
  - `RectangularComponent(width, length, x0=0, y0=0, reflectivity=1)` → `galsim.Box(w, l)`,
    `area = width * length` (mirror `RectangularOrbitalObject`, `orbital_objects.py:326-347`).
- Follow existing idioms: `@enforce_units`, `_name`-backed read-only properties, quantity type
  aliases and `Scalar` shape markers.

### Milestone 2: CompositeOrbitalObject

**Files**: `src/metroid/profiles/orbital_objects.py`

**Code Intent**:

- Add `CompositeOrbitalObject(OrbitalObject)`. Constructor (unit-enforced) takes the shared orbital
  parameters `height`, `zenith_angle`, `rotation_angle=0 deg`, `nadir_pointing=False` (forwarded to
  `super().__init__`) plus `components: Sequence[Component]`.
- Validate `components` is non-empty and every element is a `Component` (raise `TypeError`/
  `ValueError` consistent with existing style, e.g. `orbital_objects.py:217-221`). Store as an
  immutable tuple; expose read-only `components` property.
- Override `area -> Area[Scalar]`: sum of `component.area` over all components (Decision Log:
  area = sum of parts).
- Override `profile -> galsim.GSObject`:
  1. Build each component profile via `component.get_profile(self.distance)`.
  2. Combine with `galsim.Sum(*profiles)` (or `functools.reduce(operator.add, ...)`).
  3. If `self.nadir_pointing`, return `self._project(summed)`; else return `summed` — exactly
     mirroring the primitive pattern at `orbital_objects.py:278-282`. (Note: this preserves the
     current toggle behavior; the continuous-pointing-angle roadmap item, issue #42, will later
     change how/when `_project` is applied — keep the call site identical to the primitives so both
     roadmap items converge on one code path.)
- No changes to `get_tracked_profile`, `_project`, or any orbital-mechanics property — all inherited
  unchanged.

### Milestone 3: Exports and package documentation

**Files**: `src/metroid/profiles/__init__.py`, `src/metroid/profiles/README.md`,
`src/metroid/profiles/CLAUDE.md`

**Code Intent**:

- Export `Component`, `CircularComponent`, `RectangularComponent`, `CompositeOrbitalObject` from
  `profiles/__init__.py`; add to `__all__` in sorted order (match existing style).
- Update `profiles/CLAUDE.md` navigation table: add a `components.py` row and note the new
  composite class in the `orbital_objects.py` row.
- Technical Writer writes/updates `profiles/README.md` with the Invisible Knowledge above.

### Milestone 4: Tests

**Files**: `tests/profiles/test_components.py` (new),
`tests/profiles/test_orbital_objects.py` (extend)

**Code Intent**:

- `Component` unit tests: construction + unit enforcement (bad units/shape raise); `area` values for
  circular/rectangular; `reflectivity` range validation; `relative_flux()` scales with reflectivity
  and area; `get_profile` returns a `galsim.GSObject`, has the expected flux
  (`profile.flux ≈ relative_flux()`), and is shifted (centroid at expected arcsec offset).
- `CompositeOrbitalObject` tests: rejects empty/non-`Component` inputs; `area` equals the sum of
  component areas; `profile` is a `galsim.GSObject`; total composite flux equals the sum of
  component fluxes (verifies `galsim.Sum` flux preservation and relative brightness scaling);
  `get_tracked_profile` returns a `galsim.Convolution` (drop-in with primitives); with
  `nadir_pointing=True` the profile differs from the unprojected sum (projection applied once).
- Assert the shift/centroid convention explicitly (Known Risks: shift sign) — e.g. a single
  off-center component's `profile.centroid` matches the expected arcsec offset.
- Run `black`, `mypy`, `pytest` (per project CLAUDE.md tooling).
