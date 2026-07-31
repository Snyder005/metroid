# Flux Scaling for Object Profiles

## Overview

`metroid` renders satellite surface-brightness profiles whose absolute scale is arbitrary: primitive
objects return unit-flux `galsim` shapes and `CompositeOrbitalObject` sums components weighted by a
*relative* `reflectivity × area` (`profiles/components.py:67-84`). The observatory-facing brightness
quantity is a **magnitude** (what telescopes report), and nothing currently converts a magnitude into
the absolute flux a rendered profile should integrate to. This plan adds that bridge: an input
magnitude routes through the existing photometry layer
(`ThroughputCurve.calculate_adu → photon_flux_to_adu`) to a total **ADU**, which scales the object's
`galsim` profile via `withFlux`. For a `CompositeOrbitalObject`, scaling the *summed* profile to the
ADU total automatically distributes flux across components in proportion to their relative weights
(`galsim.Sum` + `withFlux` are both linear), realizing the directive "scale the total, distribute by
relative scalings" with no per-component code.

The logic lives at two layers: an `OrbitalObject` method (so a user can scale/study a profile without
building a full `Observatory`) and an `Observatory` convenience method that pulls the bandpass and
photometric parameters from its own camera/pupil and routes through the object-level method. The
feature is split into a **core** deliverable (Milestones 1–4: magnitude→flux→profile + composite
distribution + magnitude↔ADU reversibility) that fully satisfies the task directive, and an
**optional canonical-flux layer** (Milestone 5: canonical magnitude at a reference geometry, range
`1/d²` scaling, and projection flux relaxation) that implements the fuller workflow proposed in issue
#35 and is gated on an explicit decision because it mutates the shared `_project` code path. See the
companion `plans/flux-scaling-report.md` for the full investigation.

## Planning Context

This section is consumed VERBATIM by downstream agents (Technical Writer, Quality Reviewer).

### Decision Log

| Decision | Reasoning Chain |
| -------- | --------------- |
| Input is always a **magnitude**, routed through the photometry layer to flux; never scale a profile by a raw flux passed in directly. | Observatories report magnitudes, not fluxes -> the model's public brightness input must match the observable so users supply what they measure -> forcing magnitude → `ThroughputCurve.calculate_adu` → ADU as the only path keeps a single, physically-grounded conversion (`throughput.py:168-195`) and prevents callers from injecting an unnormalized flux that would bypass photometry. |
| Adopt the convention **"galsim flux ≡ total ADU"** for the scaled profile. | `galsim.GSObject.flux` is a unitless integrated-counts scalar with no intrinsic unit; GalSim only unitizes flux for chromatic SED×Bandpass objects -> metroid already carries flux as a convention (dimensionless weight in `pupils.py:207`, `reflectivity×area` m² in `components.py:113`) -> the photometry chain `ph/(s·m²)·s·(e⁻/ph)·m²/(e⁻/adu)` lands cleanly in `u.adu` (`conversions.py:26-50`), so `withFlux(adu_value)` makes the drawn profile integrate to that many ADU, matching the maintainer's existing external workflow. |
| Scale the **composite** by applying `withFlux(total_adu)` to the *summed* profile, with no per-component loop. | `galsim.Sum` preserves each summand's flux and `_project` scales the sum uniformly, so component flux *ratios* are preserved through `profile` -> `withFlux` rescales the whole object linearly to the target total, preserving those ratios -> each component ends at `total_adu · w_i / Σ w_j` (its relative share), which is exactly "distribute the scaled total by relative scalings" -> the arbitrary m² absolute scale of `relative_flux` is normalized away, resolving the composite plan's deferred-normalization assumption without touching `Component.relative_flux`. |
| Apply the ADU flux to the **convolved** profile in the tracked path, not to the bare `profile`. | Pupil defocus is not unit-flux — `AnnularPupil.get_profile` returns `TopHat(r_o) − TopHat(r_i, flux=(r_i/r_o)²)`, total flux `1−(r_i/r_o)²` (`pupils.py:207`) -> `galsim.Convolve` multiplies summand fluxes, so scaling the bare profile before convolving would corrupt the final normalization -> calling `withFlux(total_adu)` on the `galsim.Convolve` result forces the final integrated flux to exactly `total_adu` regardless of intermediate flux bookkeeping. |
| Provide **both** an `OrbitalObject`-level method and an `Observatory`-level method, the latter routing through the former. | Users may want to manipulate/study a satellite profile in isolation without constructing a full `Observatory` (per maintainer directive) -> the object level must own the bridge + scaling so it is usable standalone -> the `Observatory` already owns the bandpass catalogue (`camera[band]`) and `get_photo_params` (`observatory.py:137-161`), so its method is a thin convenience that resolves those inputs and delegates, avoiding duplicated conversion logic. |
| Add `calculate_flux` as a **separate, isolated** method from `get_scaled_profile`. | The magnitude→ADU bridge is independently useful (users may want the number, and it is the single place the photometry dependency enters `OrbitalObject`) -> isolating it keeps `get_scaled_profile` a thin "compute flux, then `withFlux`" wrapper -> mirrors how `composite-satellites.md` isolated `relative_flux` as the documented extension point for exactly this integration. |
| Reverse direction (ADU/flux → magnitude) implemented as a `ThroughputCurve` method using the linear AB relation. | Issue #35 requires the conversion to "work both ways" -> `_flux` already exploits that AB flux is linear in `10^(−0.4·mag)` (`throughput.py:246-253`), so the inverse is `mag = −2.5·log₁₀(adu/adu₀)` with `adu₀ = calculate_adu(0.0, photo_params)` -> placing it on `ThroughputCurve` co-locates it with the forward `calculate_adu` and reuses the same zeropoint machinery. |
| Take `brightness_spec: float | int | Sed` (not magnitude-only) on the object/observatory methods. | `ThroughputCurve.calculate_adu` already accepts either an AB magnitude or a full `Sed` (`throughput.py:169-195`) -> threading the same union through avoids narrowing the public API below what the photometry layer supports -> a user with a measured SED gets absolute scaling for free, and the magnitude case (the directive's focus) is the common path. |
| Defer canonical/range/projection scaling to a gated Milestone 5, not the core. | Range scaling `∝1/d²` only has meaning relative to a *canonical* reference distance; applied to an already-observed magnitude it would double-count the distance already baked into that brightness -> so range scaling requires the canonical-magnitude concept, which is a distinct feature from the magnitude→profile bridge -> and relaxing `_project`'s flux-conserving `÷mu` mutates a code path shared by all objects that only just stabilized across #42/#41, so it needs an explicit decision -> isolating it as Milestone 5 lets the core (Milestones 1–4) satisfy the directive and merge independently. |

### Rejected Alternatives

| Alternative | Why Rejected |
| ----------- | ------------ |
| Accept a raw flux (ADU or photons) as the profile-scaling input. | Contradicts the directive that input must always be a magnitude routed through photometry; lets callers bypass the physical conversion and inject an unnormalized number. |
| Loop over components and set each one's flux to its distributed share explicitly. | Unnecessary: `galsim.Sum` + `withFlux` on the summed profile already distributes linearly by relative weight. A manual loop duplicates GalSim's behavior and risks drift from the `profile` property's own summation/projection. |
| Scale the bare `profile` and then convolve with PSF/pupil. | The pupil defocus profile is not unit-flux (`pupils.py:207`), so convolution would rescale the result away from the target ADU. Must scale the convolved output. |
| Change `Component.relative_flux` to return ADU / absolute units. | Components have no orbital state, no distance, no bandpass, and no photometric parameters — absolute flux is a whole-object property. `relative_flux` is correctly a *relative* weight; `withFlux` on the composite supplies the absolute scale. |
| Put the whole workflow only on `Observatory`. | Precludes studying a satellite profile without constructing a full observatory, which the maintainer explicitly wants supported. |
| Put the whole workflow only on `OrbitalObject`. | Forces every caller to hand-resolve the bandpass `ThroughputCurve` and build `PhotometricParameters` even when they already have an `Observatory` that owns both. |
| Implement the full canonical/range/projection workflow now (single deliverable). | Range scaling needs the canonical-magnitude concept to avoid double-counting distance, and projection flux relaxation mutates the shared `_project` invariant; bundling them delays the directive-critical bridge and couples it to a gated decision. |

### Constraints & Assumptions

- **Directive (Tier 1 user instruction):** input is always a magnitude → photometry → flux → scale
  profile; the composite total is scaled and then distributed over components by relative scaling.
- **Unit convention:** metroid galsim flux ≡ total ADU (confirmed with maintainer 2026-07-31). See
  `plans/flux-scaling-report.md` and the `galsim-flux-unit-convention` memory.
- **Noiseless assumption:** ADU-vs-electrons is indistinguishable for a noiseless profile; Poisson
  noise (which is Poissonian in electrons) is out of scope. Documented, not built.
- **No import cycle:** `photometry` does not import `profiles` (verified), so `orbital_objects.py`
  may import `ThroughputCurve` and `PhotometricParameters` from `metroid.photometry`.
- **Existing patterns to follow:** `@enforce_units` on quantity-typed params/returns
  (`decorators.py`); quantity specs + shape markers from `metroid.utils.quantities` (`Adu[Scalar]`);
  `isinstance` guards with the existing error-message style (`observatory.py:19-33`,
  `orbital_objects.py:233-237`); physical→angular conversion idiom unchanged.
- **Reuse, don't reinvent:** `ThroughputCurve.calculate_adu` (`throughput.py:168-195`) and
  `photon_flux_to_adu` (`conversions.py:26-50`) already perform magnitude→ADU. This feature *calls*
  them; it does not reimplement photometry.
- **Dependencies:** `galsim` (`withFlux`, `Convolve`, `Sum`), `astropy.units`, `numpy` — all already
  used. No new third-party dependencies.
- **Package layout:** object-level code in `src/metroid/profiles/orbital_objects.py`; observatory
  orchestration in `src/metroid/observatory.py`; reverse conversion in
  `src/metroid/photometry/throughput.py`. Tests mirror under `tests/`.

### Known Risks

| Risk | Mitigation | Anchor |
| ---- | ---------- | ------ |
| Applying flux before convolution mis-normalizes, because pupil defocus is not unit-flux. | Apply `withFlux(total_adu)` to the `galsim.Convolve` result in the tracked path; scale `self.profile` directly only in the untracked `get_scaled_profile`. | `src/metroid/profiles/pupils.py:207` — `galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i / r_o) ** 2)` (total flux `1−(r_i/r_o)²`, not 1). |
| Composite distribution could be wrong if `galsim.Sum`/`withFlux`/`_project` did not preserve flux ratios. | Rely on GalSim linearity (`Sum` preserves summand flux, `withFlux`/`_project` scale uniformly); add an explicit test asserting each component's post-scale flux equals `total_adu · w_i / Σ w_j`. | `src/metroid/profiles/orbital_objects.py:394-405` — `profile` = `_project(galsim.Sum(component profiles))`; `Component.relative_flux` weights at `components.py:67-84`. |
| `profiles → photometry` import introduces a cycle. | No cycle exists: `photometry` imports only from `metroid.utils`, never `profiles` (verified by grep). Import `ThroughputCurve`/`PhotometricParameters` at module top of `orbital_objects.py`. | `src/metroid/photometry/throughput.py:1-12`, `conversions.py:1-3` — imports are from `.` and `metroid.utils` only. |
| Reverse conversion divides by a zeropoint that could be zero/degenerate. | `adu₀ = calculate_adu(0.0, photo_params)` is strictly positive for any physical bandpass (AB zeropoint > 0, positive exptime/qe/area/gain); guard against non-positive `adu` input with a `ValueError`. | `src/metroid/photometry/throughput.py:246-253` — magnitude scale is `10**(-0.4*mag)`, strictly positive. |
| Milestone 5 relaxing `_project`'s `÷ mu` changes flux for *every* object and contradicts a documented invariant. | Gate behind an explicit maintainer decision; confine to Milestone 5; keep Milestones 1–4 independent and mergeable without it. | `src/metroid/profiles/README.md` — "The `/ mu` term in `_project` conserves total flux under foreshortening; `mu > 1` would spuriously amplify flux." |
| `Observatory.get_scaled_profile` with a PSF but the object already lives at a distance inconsistent with the pupil. | No new coupling: reuse the existing `get_tracked_profile(psf, pupil)` contract unchanged; the method only supplies `self.pupil` and the resolved `photo_params`. | `src/metroid/profiles/orbital_objects.py:213-241` — `get_tracked_profile` already convolves `profile`, pupil defocus, and psf. |

## Invisible Knowledge

Technical Writer: update `src/metroid/profiles/README.md` and `src/metroid/CLAUDE.md` (and the
`observatory.py`/`throughput.py` rows). Create/extend co-located docs.

1. **Business rule — brightness input is always a magnitude, never a raw flux.** Observatories report
   magnitudes, so the only supported way to set a profile's absolute scale is a magnitude (or a full
   `Sed`) routed through `ThroughputCurve.calculate_adu`. The resulting ADU is the profile's `galsim`
   flux. This is a hard directive, not a convenience — it keeps the physical conversion in one place.

2. **Convention — metroid galsim flux ≡ total ADU.** A `galsim.GSObject.flux` is a unitless
   integrated-counts scalar. metroid interprets that scalar as **ADU** (the output of the photometry
   chain). This is a *convention*, carried by documentation and by the scaling methods, not something
   GalSim enforces. Valid today because the profile is **noiseless**; if Poisson noise is ever added
   it must be applied in electrons (`×gain`) before `/gain → ADU`, since shot noise is Poissonian in
   electrons, not ADU.

3. **System invariant — composite distribution is automatic via GalSim linearity.** Scaling a
   `CompositeOrbitalObject` to a total ADU is `summed_profile.withFlux(total_adu)`; because
   `galsim.Sum` preserves summand flux and `withFlux`/`_project` scale uniformly, each component ends
   at `total_adu · (reflectivity_i·area_i) / Σ(reflectivity_j·area_j)`. Do **not** add a per-component
   loop — it would duplicate and risk desyncing from GalSim's own bookkeeping. `Component.relative_flux`
   stays a *relative* weight; its arbitrary m² scale is normalized away by `withFlux`.

4. **System invariant — scale after convolution in the tracked path.** The pupil defocus profile is
   not unit-flux (`AnnularPupil` → `1−(r_i/r_o)²`). Flux must be applied to the convolved
   `galsim.Convolve` output, never to the bare `profile` that is then convolved, or the final
   normalization is wrong.

5. **Architectural decision — two entry points, one implementation.** The `OrbitalObject` method is
   the real implementation (usable standalone for profile studies); the `Observatory` method is a
   convenience that resolves the bandpass `ThroughputCurve` from `self.camera[band]` and the
   `PhotometricParameters` from `self.get_photo_params(exptime)`, then delegates. There is exactly one
   copy of the magnitude→flux→scale logic.

6. **Scope boundary (historical context) — observed vs. canonical magnitude.** Milestones 1–4 treat
   the input magnitude as the **observed** brightness at the actual geometry and scale the profile
   directly. Range scaling (`∝1/d²`) and projection flux relaxation are *not* applied here because
   they only make sense relative to a **canonical** reference geometry (a magnitude standardized at,
   e.g., a canonical orbital height observed at zenith); applying `1/d²` to an already-observed
   magnitude double-counts distance. That canonical workflow — the fuller proposal in issue #35 — is
   Milestone 5, gated on a decision to relax `_project`'s flux conservation.

## Milestones

### Milestone 1: Reverse conversion — ADU ↔ magnitude on `ThroughputCurve`

**Files**: `src/metroid/photometry/throughput.py`, `tests/photometry/test_throughput.py`

**Code Intent**:

- Add `calculate_ab_magnitude_from_adu(adu: Adu[Scalar], photo_params: PhotometricParameters) -> float`
  to `ThroughputCurve`. Compute the zero-magnitude reference `adu0 = self.calculate_adu(0.0, photo_params)`
  and return `-2.5 * log10((adu / adu0).to_value(u.dimensionless_unscaled))`. This inverts the linear
  AB relation the forward path already uses (Decision Log: reverse conversion).
- Decorate with `@enforce_units` for the `adu` parameter (`Adu[Scalar]`); guard `photo_params` type
  with the existing `isinstance` style (`throughput.py:47-48`); raise `ValueError` for non-positive
  `adu`.
- Docstring in the existing NumPy style; note that AB magnitudes are unitless `float` (mirror the note
  at `throughput.py:210-215`).
- Satisfies issue #35's "work both ways" requirement at the photometry layer, independent of the
  profile changes.

**Tests**: round-trip `calculate_adu(m) → calculate_ab_magnitude_from_adu → m`; monotonicity
(brighter/smaller magnitude ⇒ larger ADU); `ValueError` on non-positive ADU; unit enforcement on
`adu`.

### Milestone 2: Object-level flux bridge and scaled profiles on `OrbitalObject`

**Files**: `src/metroid/profiles/orbital_objects.py`, `tests/profiles/test_orbital_objects.py`

**Code Intent**:

- Import `ThroughputCurve` and `PhotometricParameters` from `metroid.photometry` at module top (no
  cycle — see Known Risks).
- Add three methods to the `OrbitalObject` ABC (all inherited by every subclass, incl.
  `CompositeOrbitalObject`):
  - `calculate_flux(brightness_spec: float | int | Sed, throughput: ThroughputCurve, photo_params: PhotometricParameters) -> Adu[Scalar]`
    — the isolated bridge: guard `throughput`/`photo_params` types, return
    `throughput.calculate_adu(brightness_spec, photo_params)`. This is the single place the photometry
    dependency enters (Decision Log: isolated bridge).
  - `get_scaled_profile(brightness_spec, throughput, photo_params) -> galsim.GSObject` — compute
    `total_adu = self.calculate_flux(...)` and return `self.profile.withFlux(total_adu.to_value(u.adu))`.
    For studying a bare (untracked) profile at absolute scale.
  - `get_scaled_tracked_profile(brightness_spec, throughput, photo_params, psf: galsim.GSObject, telescope_pupil: Pupil) -> galsim.Convolution`
    — build `tracked = self.get_tracked_profile(psf, telescope_pupil)` then return
    `tracked.withFlux(total_adu.to_value(u.adu))`. Flux applied **after** convolution (Decision Log /
    Known Risks: pupil defocus non-unit-flux).
- Reuse `get_tracked_profile`'s existing type guards; do not duplicate them where delegation covers it.
- NumPy-style docstrings; `@enforce_units` where a parameter/return is a quantity type.

**Tests** (extend existing file):
- `calculate_flux` equals `throughput.calculate_adu` for both a magnitude and an `Sed`.
- `get_scaled_profile(...).flux ≈ calculate_flux(...).to_value(u.adu)` for `CircularOrbitalObject` and
  `RectangularOrbitalObject`.
- `get_scaled_tracked_profile(...)` returns a `galsim.Convolution` whose `.flux` ≈ the ADU total
  (verifies post-convolution scaling survives a non-unit-flux `AnnularPupil` defocus).
- Brighter magnitude ⇒ larger profile flux (monotonic, linear in `10^(−0.4·mag)`).

### Milestone 3: Composite distribution behavior (tests + docs only)

**Files**: `tests/profiles/test_orbital_objects.py`, `src/metroid/profiles/README.md`

**Code Intent**:

- **No new production code** — `CompositeOrbitalObject` inherits Milestone 2's methods, and
  `withFlux` on the summed profile distributes automatically (Decision Log: composite distribution).
- Add tests asserting the distribution invariant: build a `CompositeOrbitalObject` with two components
  of known `reflectivity × area` weights `w_1, w_2`; after `get_scaled_profile(magnitude, ...)`, assert
  total flux ≈ `total_adu` and that the per-component flux ratio is preserved as `w_1 : w_2`
  (recover per-component contributions by summing the fluxes of the `galsim.Sum` obj_list, or by
  comparing against `total_adu · w_i / Σ w_j`).
- Assert `get_scaled_tracked_profile` on a composite also integrates to `total_adu`.
- Update `profiles/README.md` with Invisible Knowledge items 2–4 and 6 (convention, automatic
  distribution, scale-after-convolution, observed-vs-canonical boundary).

### Milestone 4: Observatory orchestration

**Files**: `src/metroid/observatory.py`, `tests/test_observatory.py`

**Code Intent**:

- Add `get_scaled_profile(self, orbital_object: OrbitalObject, brightness_spec: float | int | Sed, band: str, exptime: Time[Scalar], psf: galsim.GSObject | None = None) -> galsim.GSObject`.
  - Guard `orbital_object` is an `OrbitalObject` (existing error-message style).
  - Resolve `throughput = self.camera[band]` (raises `ValueError` for unknown band via
    `Camera.__getitem__`, `camera.py:140-144`) and `photo_params = self.get_photo_params(exptime)`
    (`observatory.py:137-161`).
  - If `psf is None`: return `orbital_object.get_scaled_profile(brightness_spec, throughput, photo_params)`.
  - Else: return `orbital_object.get_scaled_tracked_profile(brightness_spec, throughput, photo_params, psf, self.pupil)`.
- Thin convenience only — one delegation each way, no duplicated conversion logic (Decision Log: two
  entry points, one implementation).
- Import `OrbitalObject` from `metroid.profiles`; `@enforce_units` for `exptime`.

**Tests**: `get_scaled_profile` with and without a `psf` produces flux ≈ the ADU total; matches the
result of calling the object-level method directly with the manually-resolved `throughput`/`photo_params`
(proves routing); unknown `band` raises `ValueError`; non-`OrbitalObject` raises.

### Milestone 5 (GATED — optional canonical-flux workflow): canonical magnitude, range & projection scaling

**Files**: `src/metroid/profiles/orbital_objects.py`, `src/metroid/photometry/conversions.py`
(possibly), tests

**Status**: **Do not implement without an explicit maintainer decision.** This milestone relaxes the
flux-conserving `÷ mu` in `_project` and introduces a canonical reference geometry — a superset of the
core feature and a change to a shared code path (see Known Risks). Milestones 1–4 are complete and
mergeable without it.

**Code Intent** (specification for when unblocked):

- **Canonical magnitude concept.** Define a canonical reference geometry (candidate per issue #35: a
  canonical orbital height observed at zenith → `distance_canonical`). Interpret the input magnitude
  as the brightness at that canonical geometry, not the observed one.
- **Range scaling.** Multiply the canonical flux by `(distance_canonical / self.distance) ** 2` before
  `withFlux`, so an object farther than canonical is dimmer. (Only valid relative to the canonical
  reference — see Invisible Knowledge item 6.)
- **Projection flux scaling.** Optionally relax `_project`'s `÷ mu` so foreshortening changes total
  flux (the issue's "relax the flux-preserving restriction"). This must be a deliberate, documented
  toggle because it changes behavior for *all* objects and contradicts the current README invariant.
- **Reversibility.** Provide observed↔canonical flux conversion (derive canonical flux from an observed
  flux by inverting the range/projection terms), completing issue #35's "should work both ways" at the
  geometric level.
- **BRDF.** Treat as fully diffuse / angle-independent (issue #35 explicit scope): reflectivity is a
  linear radiance scaling only; no incidence/scattering-angle dependence yet.

**Decision required before starting**: whether to relax `_project` flux conservation (and if so,
default on or off), and the exact definition of the canonical reference geometry.

## Finalization

- Run `black`, `mypy`, `pytest` (project CLAUDE.md tooling) after each milestone.
- Milestones 1–4 constitute the reviewable core PR for issue #35; Milestone 5 is deferred pending the
  gating decision (split into a follow-up issue/PR if approved).
- Draft PR opened early per the Git Workflow; `Fixes #35` in the PR body (core scope; note Milestone 5
  deferral).
