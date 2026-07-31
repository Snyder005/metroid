# Flux Scaling for Object Profiles

## Overview

`metroid` renders satellite surface-brightness profiles whose absolute scale is arbitrary: primitive
objects return unit-flux `galsim` shapes and `CompositeOrbitalObject` sums components weighted by a
*relative* `reflectivity × area` (`profiles/components.py:67-84`). The observatory-facing brightness
quantity is a **magnitude** (what telescopes report), and nothing currently converts a magnitude into
the absolute flux a rendered profile should integrate to. This plan adds that bridge and the physics
around it.

The construction-time magnitude is the **observed** AB magnitude at the object's actual geometry. It
is immediately converted to a geometry-invariant **canonical magnitude** — the brightness the object
would have at a canonical reference geometry of **500 km height observed at zenith** — and stored,
because `height`/`zenith_angle`/`pointing_angle` are mutable and a stored *observed* magnitude would go
stale. The observed magnitude for the current geometry is re-derived on demand. At render time the
observed magnitude routes through the existing photometry layer (`ThroughputCurve.calculate_adu`) to a
total **ADU** (metroid convention: galsim flux ≡ total ADU), which scales the profile via `withFlux`
applied to the **convolved** output (normalize to 1.0, then scale — necessary because pupil defocus is
not unit-flux). For a composite, scaling the summed profile distributes flux across components in
proportion to their relative weights (GalSim linearity), realizing "scale the total, distribute by
relative scalings" with no per-component loop.

Two physics decisions accompany the bridge. First, `OrbitalObject._project`'s flux-conserving `÷ mu`
is **relaxed** (dropped) so foreshortening dims total flux by the projected-area factor `mu` — matching
Lumos-Sat's `observer_normalization` and issue #35's "relax the flux-preserving restriction." Second, a
**projection seam** is laid: `CompositeOrbitalObject.profile` projects **each component** (with a single
shared angle) and then sums, instead of projecting the sum. This is provably output-identical today but
lets a future independent/3D component-orientation model make projection component-aware without
restructuring `orbital_objects.py`. See `plans/flux-scaling-report.md` for the full derivation and
Lumos-Sat cross-check.

## Planning Context

This section is consumed VERBATIM by downstream agents (Technical Writer, Quality Reviewer).

### Decision Log

| Decision | Reasoning Chain |
| -------- | --------------- |
| Brightness input is always a **magnitude** routed through photometry; never a raw flux. | Observatories report magnitudes, not fluxes -> the public brightness input must match the observable so users supply what they measure -> forcing magnitude → `ThroughputCurve.calculate_adu` → ADU as the only path keeps one physically-grounded conversion (`throughput.py:168-195`) and prevents callers injecting an unnormalized flux that bypasses photometry. |
| Adopt the convention **galsim flux ≡ total ADU**. | `galsim.GSObject.flux` is a unitless integrated-counts scalar with no intrinsic unit; GalSim unitizes flux only for chromatic SED×Bandpass objects -> metroid already carries flux as a convention (dimensionless weight `pupils.py:207`; `reflectivity×area` m² `components.py:113`) -> the photometry chain `ph/(s·m²)·s·(e⁻/ph)·m²/(e⁻/adu)` lands cleanly in `u.adu` (`conversions.py:26-50`) -> `withFlux(adu_value)` makes the drawn profile integrate to that many ADU, matching the maintainer's existing external workflow. |
| Interpret the construction magnitude as **observed** and store the derived **canonical** magnitude (500 km, zenith). | The brightness a telescope reports is at the actual geometry, so the input is naturally the observed magnitude -> but `height`/`zenith_angle`/`pointing_angle` are mutable setters (`orbital_objects.py:52-105`), so a stored observed magnitude would silently go stale when geometry changes -> the canonical magnitude is geometry-invariant by definition, so storing it (and re-deriving observed on demand) is the only representation robust to mutation -> matches the maintainer's directive to expose the 500 km magnitude as an object property. |
| Observed↔canonical conversion is **purely geometric** (no photometry). | Both magnitudes describe the same source, so their difference is a flux *ratio* -> flux ∝ `mu / distance²` (projection foreshortening × inverse-square range), and a ratio cancels all band/photometry terms -> therefore `m_can = m_obs + 2.5·log₁₀(mu_obs) − 5·log₁₀(d_obs/d_can)` is band-independent and cacheable, and its inverse re-derives observed at any geometry -> satisfies issue #35 "work both ways" without threading a bandpass through construction. |
| **Relax** `_project`'s flux-conserving `÷ mu` (drop it). | `galsim`'s `transform(mu,0,0,1)` scales flux by the Jacobian `mu` (verified: TopHat flux 1.0→0.5 at mu=0.5); the current `÷ mu` restores it to conserve flux -> issue #35 asks to relax that restriction so foreshortening dims flux by projected area, matching Lumos' `observer_normalization = clip(normal·obs_dir,0)` -> dropping `÷ mu` is output-neutral for *scaled* profiles today (`withFlux` overrides the total; a shared-angle composite's common `mu` cancels in `wᵢ/Σwⱼ`) and only becomes observable per-component in the future 3D model -> so relax now to avoid a flux-semantics change later, accepting a bare-`profile` flux change (1.0→mu) and a README/test update. |
| **Lay the per-component projection seam now**: project each component (shared angle) then sum. | `_project` is a linear transform about the body-frame origin and `galsim.Sum` is linear, so `project(Σ shiftᵢ(Pᵢ)) ≡ Σ project(shiftᵢ(Pᵢ))` -> relocating projection into a per-component step is provably identical while the composite drives one shared angle -> the future independent/3D-orientation roadmap needs projection to be component-aware (each component's own `mu` from its own normal), and laying the structural seam now means that model changes only the projection call, not the composite's structure -> pairs with the `÷ mu` relaxation so differing per-component `muᵢ` later yield correct projected-area weighting. |
| Composite distribution via `withFlux(total_adu)` on the summed/projected profile — no per-component loop. | `galsim.Sum` preserves each summand's flux and projection scales uniformly (shared angle), so component flux *ratios* survive into `profile` -> `withFlux` rescales the whole object linearly to the target total, preserving those ratios -> each component ends at `total_adu · wᵢ/Σⱼwⱼ`, exactly "distribute the scaled total by relative scalings" -> the arbitrary m² scale of `relative_flux` is normalized away, resolving the composite plan's deferred-normalization assumption without touching `Component.relative_flux`. |
| Apply the ADU flux to the **convolved** profile in the tracked path (normalize to 1.0, then scale). | Pupil defocus is not unit-flux — `AnnularPupil.get_profile` returns `TopHat(r_o) − TopHat(r_i, flux=(r_i/r_o)²)`, total flux `1−(r_i/r_o)²` (`pupils.py:207`) -> `galsim.Convolve` multiplies summand fluxes, so scaling the bare profile before convolving corrupts the final normalization -> `withFlux` divides by current flux and multiplies by target, so calling it on the `galsim.Convolve` result sets the final integrated flux to exactly `total_adu` regardless of intermediate bookkeeping. |
| Provide **both** an `OrbitalObject` method and an `Observatory` method, the latter routing through the former. | Users may study a satellite profile in isolation without a full `Observatory` (maintainer directive) -> the object level must own the bridge + scaling to be usable standalone -> `Observatory` already owns the bandpass catalogue (`camera[band]`) and `get_photo_params` (`observatory.py:137-161`), so its method is a thin convenience that resolves those and delegates, keeping one copy of the conversion logic. |
| Isolate `calculate_flux` from `get_scaled_profile`. | The magnitude→ADU bridge is independently useful and is the single point where the photometry dependency enters `OrbitalObject` -> isolating it keeps `get_scaled_profile` a thin "compute flux, then `withFlux`" wrapper -> mirrors how the composite plan isolated `relative_flux` as the documented extension point for this integration. |
| Reverse conversion (ADU→magnitude) as a `ThroughputCurve` method. | Issue #35 requires the conversion to "work both ways" -> `_flux` already exploits AB flux being linear in `10^(−0.4·mag)` (`throughput.py:246-253`), so the inverse is `mag = −2.5·log₁₀(adu/adu₀)` with `adu₀ = calculate_adu(0.0, photo_params)` -> placing it on `ThroughputCurve` co-locates it with forward `calculate_adu` and reuses the zeropoint machinery. |

### Rejected Alternatives

| Alternative | Why Rejected |
| ----------- | ------------ |
| Accept a raw flux (ADU/photons) as the profile-scaling input. | Contradicts the directive that input is always a magnitude routed through photometry; lets callers bypass the physical conversion. |
| Store the **observed** magnitude (not canonical). | `height`/`zenith_angle`/`pointing_angle` are mutable; a stored observed magnitude goes stale on any geometry change. The canonical magnitude is invariant and the robust thing to store. |
| Compute the observed↔canonical conversion through the photometry layer. | It is a flux *ratio*, so all band terms cancel — a purely geometric `mu`/`distance²` expression is exact, band-independent, and needs no `ThroughputCurve` at construction. |
| Keep `_project`'s flux-conserving `÷ mu`. | Issue #35 explicitly asks to relax it; keeping it blocks future per-component projected-area flux weighting and forces a flux-semantics change in the 3D roadmap instead of now (when it is output-neutral). |
| Keep project-once-on-the-sum and defer the seam entirely. | The future independent/3D-orientation model would then need a structural refactor of `CompositeOrbitalObject.profile`; laying the (identical-output) per-component seam now avoids that, per maintainer direction. |
| Make `_project` component-aware (per-component `mu` from normals) now. | No per-component normals exist yet; that is the far-off 3D roadmap. This plan only lays the structural seam with a single shared angle — full component-aware projection is explicitly future work. |
| Loop over components setting each one's absolute flux explicitly. | Unnecessary: `galsim.Sum` + `withFlux` distribute linearly by relative weight. A manual loop duplicates GalSim and risks drift from `profile`'s own summation. |
| Change `Component.relative_flux` to absolute units. | Components have no orbital state, distance, or bandpass — absolute flux is a whole-object property. `relative_flux` is correctly relative; `withFlux` supplies the absolute scale. |

### Constraints & Assumptions

- **Directive (Tier 1):** input is always a magnitude → photometry → flux → scale profile; canonical
  reference is 500 km at zenith exposed as an object property; the observed magnitude is the
  construction input; relax `÷ mu`; lay the projection seam now with a future-consideration note; scale
  only after convolution (set convolution to 1.0 then scale).
- **Unit convention:** galsim flux ≡ total ADU. See `plans/flux-scaling-report.md` and the
  `galsim-flux-unit-convention` memory.
- **Canonical geometry:** `d_can = 500 km`, `mu_can = 1` (zenith). `mu = cos(nadir_angle − pointing_angle)`.
- **Noiseless assumption:** ADU-vs-electrons is indistinguishable without Poisson noise (which is in
  electrons); noise is out of scope. Documented, not built.
- **BRDF:** fully diffuse / angle-independent (issue #35 scope). Reflectivity is a linear radiance
  scaling; no incidence/scattering-angle dependence.
- **Future roadmap (out of scope, seam anticipates it):** independent/3D component orientation with
  per-component normals; `lumos.geometry.Surface`-style per-surface normal/BRDF properties on
  `OrbitalObject`/`Component` (shared area etc.). Flagged, not built.
- **No import cycle:** `photometry` does not import `profiles` (verified), so `orbital_objects.py` may
  import `ThroughputCurve` and `PhotometricParameters` from `metroid.photometry`.
- **Existing patterns:** `@enforce_units` on quantity params/returns (`decorators.py`); specs + shape
  markers from `metroid.utils.quantities` (`Adu[Scalar]`); `isinstance` guards with the existing
  error-message style (`observatory.py:19-33`, `orbital_objects.py:233-237`); physical→angular
  conversion idiom unchanged. AB magnitudes are unitless `float` (per `throughput.py:210-215`).
- **Dependencies:** `galsim` (`withFlux`, `Convolve`, `Sum`, `transform`), `astropy.units`, `numpy` —
  all already used. No new third-party dependencies.
- **Layout:** object-level + projection changes in `src/metroid/profiles/orbital_objects.py`;
  orchestration in `src/metroid/observatory.py`; reverse conversion in
  `src/metroid/photometry/throughput.py`. Tests mirror under `tests/`.

### Known Risks

| Risk | Mitigation | Anchor |
| ---- | ---------- | ------ |
| Applying flux before convolution mis-normalizes (pupil defocus is not unit-flux). | Apply `withFlux(total_adu)` to the `galsim.Convolve` result; scale `self.profile` directly only in the untracked `get_scaled_profile`. Test that a composite + `AnnularPupil` tracked profile still integrates to the ADU total. | `src/metroid/profiles/pupils.py:207` — `galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i / r_o) ** 2)` (total flux `1−(r_i/r_o)²`). |
| Relaxing `÷ mu` changes bare-`profile` flux for every object and contradicts a documented invariant. | Intended per directive. Update `profiles/README.md` (remove "÷mu conserves flux"; state foreshortening dims flux by `mu`) and any bare-profile flux test. Verify scaled-profile results are unchanged (withFlux overrides). | `src/metroid/profiles/orbital_objects.py:257-260` — `.transform(mu,0,0,1)...  / mu`; `src/metroid/profiles/README.md` invariant text. |
| Per-component projection seam accidentally changes composite output. | The transform is linear and `galsim.Sum` is linear, so project-then-sum ≡ project-the-sum for a shared angle. Add a regression test asserting the composite profile (moments/flux) is unchanged versus the previous project-on-sum result. | `src/metroid/profiles/orbital_objects.py:394-405` — current `self._project(galsim.Sum(profiles))`. |
| Composite distribution wrong if `Sum`/`withFlux`/`_project` didn't preserve ratios. | Rely on GalSim linearity; test each component's post-scale flux equals `total_adu · wᵢ/Σⱼwⱼ`. | `Component.relative_flux` weights `profiles/components.py:67-84`. |
| `profiles → photometry` import cycle. | No cycle: `photometry` imports only from `metroid.utils`, never `profiles` (verified). Import at module top of `orbital_objects.py`. | `src/metroid/photometry/throughput.py:1-12`, `conversions.py:1-3`. |
| Canonical conversion divides by zero at `mu = 0` or degenerate geometry. | `mu = cos(nadir_angle − pointing_angle)` with `pointing_angle ∈ [0, nadir_angle]` gives `mu ∈ [cos(nadir_angle), 1]`, strictly > 0 for physical geometries; `log₁₀(mu)` is finite. Guard `magnitude=None` (no scaling) explicitly. | `src/metroid/profiles/orbital_objects.py:96-105` — pointing_angle range validation. |
| Reverse ADU→magnitude divides by a degenerate zeropoint. | `adu₀ = calculate_adu(0.0, photo_params)` is strictly positive (AB zeropoint > 0, positive exptime/qe/area/gain); raise `ValueError` for non-positive `adu` input. | `src/metroid/photometry/throughput.py:246-253` — scale `10**(-0.4*mag)` > 0. |

## Invisible Knowledge

Technical Writer: update `src/metroid/profiles/README.md`, `src/metroid/CLAUDE.md` (rows for
`orbital_objects.py`, `observatory.py`, `throughput.py`), and `src/metroid/photometry/CLAUDE.md`.

1. **Business rule — brightness input is always a magnitude.** The only supported way to set a
   profile's absolute scale is an observed AB magnitude (or `Sed`) routed through
   `ThroughputCurve.calculate_adu`; the resulting ADU is the profile's galsim flux. A hard directive.

2. **Convention — galsim flux ≡ total ADU.** `GSObject.flux` is a unitless integrated-counts scalar;
   metroid interprets it as ADU. Carried by documentation and the scaling methods, not enforced by
   GalSim. Valid because the profile is **noiseless**; if Poisson noise is ever added it must be
   applied in electrons (`×gain`) before `/gain → ADU`, since shot noise is Poissonian in electrons.

3. **Canonical magnitude (500 km, zenith) is stored; observed is derived.** The construction magnitude
   is the *observed* brightness at the actual geometry; it is converted to the geometry-invariant
   canonical magnitude `m_can = m_obs + 2.5·log₁₀(mu_obs) − 5·log₁₀(d_obs/d_can)` and stored. This is
   purely geometric (a flux ratio, band-independent). Storing canonical — not observed — is essential
   because `height`/`zenith_angle`/`pointing_angle` are mutable; observed is re-derived per geometry.
   `flux ∝ mu / distance²` (projection × inverse-square range).

4. **System invariant — projection dims flux by `mu` (relaxed).** `_project` no longer divides by `mu`,
   so foreshortening reduces total flux by the projected-area factor `mu = cos(nadir_angle −
   pointing_angle)` — matching a diffuse Lambertian surface seen off-normal. (Previously `÷ mu`
   conserved flux; that invariant is removed.) For *scaled* profiles this is invisible because
   `withFlux` sets the final total; it matters for bare `profile` flux and for future per-component
   projected-area weighting.

5. **System invariant — composite distribution is automatic.** Scaling a `CompositeOrbitalObject` to a
   total ADU is `withFlux(total_adu)` on the summed/projected profile; each component ends at
   `total_adu · (reflectivityᵢ·areaᵢ)/Σⱼ(reflectivityⱼ·areaⱼ)`. Do **not** add a per-component flux
   loop. `Component.relative_flux` stays a relative weight; its m² scale is normalized away.

6. **System invariant — scale after convolution.** Pupil defocus is not unit-flux (`AnnularPupil` →
   `1−(r_i/r_o)²`). Flux is applied to the convolved output (`withFlux` normalizes to 1.0 then scales
   to the target), never to the bare profile that is then convolved.

7. **Architectural — the projection seam and when to make it component-aware.**
   `CompositeOrbitalObject.profile` projects **each component** with a single shared angle, then sums
   (`galsim.Sum([self._project(c.get_profile(distance)) for c in components])`), rather than projecting
   the sum. This is output-identical to project-once-on-the-sum today (both `_project` and `Sum` are
   linear). It exists so a future model can make projection **per-component**: promote the shared-angle
   `self._project(...)` to each component computing its own `mu` from its own body-frame normal + the
   shared viewing geometry. **Revisit the seam when any of:** (1) components gain independent
   orientation/pointing (own normals), (2) a 3D body is projected to a 2D observed profile, or (3)
   per-component projected-area flux weighting is required (e.g. an edge-on deployed solar panel). This
   also anticipates `lumos.geometry.Surface`-style per-surface normal/BRDF properties, a separate
   future roadmap item.

8. **Architectural — two entry points, one implementation.** The `OrbitalObject` methods are the real
   implementation (usable standalone for profile studies); `Observatory.get_scaled_profile` resolves
   the bandpass (`self.camera[band]`) and `PhotometricParameters` (`self.get_photo_params(exptime)`)
   and delegates. Exactly one copy of the magnitude→flux→scale logic.

## Milestones

### Milestone 1: Reverse conversion — ADU → magnitude on `ThroughputCurve`

**Files**: `src/metroid/photometry/throughput.py`, `tests/photometry/test_throughput.py`

**Code Intent**:

- Add `calculate_ab_magnitude_from_adu(adu: Adu[Scalar], photo_params: PhotometricParameters) -> float`.
  Compute `adu0 = self.calculate_adu(0.0, photo_params)` and return
  `-2.5 * np.log10((adu / adu0).to_value(u.dimensionless_unscaled))` (Decision Log: reverse conversion).
- `@enforce_units` on `adu` (`Adu[Scalar]`); guard `photo_params` with the existing `isinstance` style
  (`throughput.py:47-48`); raise `ValueError` for non-positive `adu`. NumPy-style docstring; note AB
  magnitudes are unitless `float` (mirror `throughput.py:210-215`).

**Tests**: round-trip `calculate_adu(m) → calculate_ab_magnitude_from_adu → m`; monotonicity (smaller
magnitude ⇒ larger ADU); `ValueError` on non-positive ADU; unit enforcement on `adu`.

### Milestone 2: Canonical magnitude on `OrbitalObject`

**Files**: `src/metroid/profiles/orbital_objects.py`, `tests/profiles/test_orbital_objects.py`

**Code Intent**:

- Add a module-level constant `CANONICAL_HEIGHT = 500.0 * u.km` (canonical geometry: this height at
  zenith, so `d_can = CANONICAL_HEIGHT`, `mu_can = 1`).
- Add optional `magnitude: float | None = None` to `OrbitalObject.__init__` and each concrete
  subclass constructor (`CircularOrbitalObject`, `RectangularOrbitalObject`, `CompositeOrbitalObject`),
  forwarded to `super().__init__`. `None` = no absolute scaling (current behavior).
- On construction (when `magnitude is not None`) convert observed→canonical and store
  `self._canonical_magnitude`: `m_can = magnitude + 2.5*log10(mu_obs) - 5*log10(d_obs/d_can)` where
  `mu_obs = cos(nadir_angle - pointing_angle)` (dimensionless), `d_obs = self.distance`,
  `d_can = CANONICAL_HEIGHT`. Use `.to_value` for the distance ratio (Decision Log: purely geometric).
- Read-only properties:
  - `canonical_magnitude -> float | None` — the stored canonical magnitude.
  - `observed_magnitude -> float | None` — re-derive for the *current* geometry:
    `m_obs = m_can - 2.5*log10(mu) + 5*log10(distance/d_can)`; returns `None` if unset.
- Add a private helper `_observed_to_canonical(mag) / _canonical_to_observed(mag)` (or inline) so the
  two conversions share one formula. NumPy-style docstrings.

**Tests**: observed→canonical→observed round-trip at the construction geometry returns the input;
`canonical_magnitude` is invariant when `height`/`zenith_angle`/`pointing_angle` are mutated while
`observed_magnitude` changes accordingly; at 500 km + zenith, `canonical == observed == input`;
`magnitude=None` leaves both properties `None`.

### Milestone 3: Object-level flux bridge and scaled profiles

**Files**: `src/metroid/profiles/orbital_objects.py`, `tests/profiles/test_orbital_objects.py`

**Code Intent**:

- Import `ThroughputCurve`, `PhotometricParameters`, and `Sed` from `metroid.photometry` at module top
  (no cycle — Known Risks).
- Add to the `OrbitalObject` ABC (inherited by all subclasses):
  - `calculate_flux(throughput, photo_params, brightness_spec=None) -> Adu[Scalar]` — the isolated
    bridge. `brightness_spec` defaults to `self.observed_magnitude` (raise `ValueError` if both are
    `None`); guard `throughput`/`photo_params` types; return
    `throughput.calculate_adu(brightness_spec, photo_params)`.
  - `get_scaled_profile(throughput, photo_params, brightness_spec=None) -> galsim.GSObject` — return
    `self.profile.withFlux(self.calculate_flux(...).to_value(u.adu))` (untracked; for profile studies).
  - `get_scaled_tracked_profile(throughput, photo_params, psf, telescope_pupil, brightness_spec=None) -> galsim.Convolution`
    — `tracked = self.get_tracked_profile(psf, telescope_pupil)`; return
    `tracked.withFlux(total_adu.to_value(u.adu))`. Flux applied **after** convolution (Decision Log /
    Known Risks). Reuse `get_tracked_profile`'s existing type guards.

**Tests**: `calculate_flux` equals `throughput.calculate_adu` for a magnitude and an `Sed`, and uses
`observed_magnitude` when `brightness_spec` omitted; `get_scaled_profile(...).flux ≈ calculate_flux`
for circular/rectangular; `get_scaled_tracked_profile(...)` returns a `galsim.Convolution` whose
`.flux ≈ ADU` even through a non-unit-flux `AnnularPupil` defocus; brighter magnitude ⇒ larger flux
(linear in `10^(−0.4·mag)`); `ValueError` when no magnitude available.

### Milestone 4: Relax projection flux and lay the per-component projection seam

**Files**: `src/metroid/profiles/orbital_objects.py`, `src/metroid/profiles/README.md`,
`tests/profiles/test_orbital_objects.py`

**Code Intent**:

- **Relax `_project`**: remove the trailing `/ mu` in `_project`
  (`orbital_objects.py:257-260`) so it returns `profile.rotate(phi).transform(mu, 0.0, 0.0, 1.0).rotate(-phi)`.
  Foreshortening now dims total flux by `mu` (Decision Log: relax `÷ mu`).
- **Per-component projection seam** in `CompositeOrbitalObject.profile`: change from
  `self._project(galsim.Sum(profiles))` to
  `galsim.Sum([self._project(c.get_profile(self.distance)) for c in self.components])` — project each
  component with the shared object-level angle, then sum (Decision Log: lay the seam now). Add an
  in-code comment recording the future-consideration note (Invisible Knowledge item 7): when to promote
  to per-component-normal projection.
- Primitive `profile` methods (`CircularOrbitalObject`, `RectangularOrbitalObject`) keep calling
  `self._project(...)` unchanged — they now inherit the relaxed (non-conserving) semantics automatically.
- Update `profiles/README.md`: replace the "`÷ mu` conserves total flux / `mu > 1` amplifies flux"
  invariant with the relaxed semantics (Invisible Knowledge item 4) and document the seam (item 7).

**Tests**: bare `CircularOrbitalObject.profile.flux ≈ mu` (relaxed, no longer 1.0) for a
foreshortened geometry, and `≈ 1.0` at zenith (`mu = 1`); composite `profile` moments/flux unchanged
versus a project-once-on-the-sum reference construction (seam is output-identical); a *scaled* profile
flux is unchanged by the relaxation (withFlux overrides).

### Milestone 5: Composite distribution behavior (tests + docs only)

**Files**: `tests/profiles/test_orbital_objects.py`, `src/metroid/profiles/README.md`

**Code Intent**:

- **No new production code** — `CompositeOrbitalObject` inherits Milestone 3's methods; `withFlux` on
  the summed/projected profile distributes automatically (Decision Log: composite distribution).
- Tests: two components of known weights `w1, w2`; after `get_scaled_profile`, assert total flux ≈
  `total_adu` and per-component contributions are `total_adu · wᵢ/Σⱼwⱼ` (recover via the `galsim.Sum`
  `obj_list` fluxes); `get_scaled_tracked_profile` on a composite also integrates to `total_adu`.
- Update `profiles/README.md` with Invisible Knowledge items 2, 5, 6.

### Milestone 6: Observatory orchestration

**Files**: `src/metroid/observatory.py`, `tests/test_observatory.py`

**Code Intent**:

- Add `get_scaled_profile(self, orbital_object: OrbitalObject, band: str, exptime: Time[Scalar], brightness_spec: float | int | Sed | None = None, psf: galsim.GSObject | None = None) -> galsim.GSObject`.
  - Guard `orbital_object` is an `OrbitalObject` (existing error-message style).
  - `throughput = self.camera[band]` (unknown band raises `ValueError` via `Camera.__getitem__`,
    `camera.py:140-144`); `photo_params = self.get_photo_params(exptime)` (`observatory.py:137-161`).
  - `psf is None` → `orbital_object.get_scaled_profile(throughput, photo_params, brightness_spec)`;
    else → `orbital_object.get_scaled_tracked_profile(throughput, photo_params, psf, self.pupil, brightness_spec)`.
- Thin convenience only, one delegation each way (Decision Log: two entry points, one implementation).
  Import `OrbitalObject` and `Sed` from `metroid.profiles` / `metroid.photometry`; `@enforce_units` for
  `exptime`.

**Tests**: with and without a `psf`, flux ≈ ADU total and matches the object-level method called with
manually-resolved `throughput`/`photo_params` (proves routing); unknown `band` raises `ValueError`;
non-`OrbitalObject` raises.

## Finalization

- Run `black`, `mypy`, `pytest` (project CLAUDE.md tooling) after each milestone.
- All six milestones are in scope for issue #35. `lumos.Surface`-style per-surface normal/BRDF
  properties and true independent/3D component orientation remain future roadmap items; the Milestone 4
  seam is laid to receive them.
- Draft PR opened early per the Git Workflow; `Fixes #35` in the PR body.
