# Flux Scaling — Implementation Report (Issue #35)

## Purpose

This report accompanies `plans/flux-scaling.md`. It records the design investigation behind the
flux-scaling feature: the physics, the GalSim/photometry unit reconciliation, the chosen API shape,
and the scope boundary between the directive-critical core and the fuller canonical-flux workflow
proposed in issue #35. The plan is the implementation directive; this report is the *why*.

## Problem

`metroid` builds a satellite surface-brightness profile (`OrbitalObject.profile`) whose absolute
scale is arbitrary:

- `CircularOrbitalObject` / `RectangularOrbitalObject` return a `galsim.TopHat` / `galsim.Box` whose
  flux defaults to unit flux, then a projection transform is applied.
- `CompositeOrbitalObject` sums component profiles whose per-component flux is
  `reflectivity × area` in m² (`Component.relative_flux`, `profiles/components.py:67-84`) — a
  deliberately **relative** weight. The composite plan (`plans/composite-satellites.md`) states this
  explicitly: "This is a *relative* model only; absolute photometric normalization (magnitude → ADU)
  is intentionally deferred to the flux-scaling roadmap (issue #35)."

The observatory-facing quantity is a **magnitude**, because that is what a telescope reports. Nothing
in the pipeline currently converts a magnitude into the absolute flux that a rendered profile should
integrate to. Issue #35 closes that gap.

## Investigation: what unit is "GalSim flux"?

A `galsim.GSObject.flux` is a **unitless scalar**: the total integrated counts the profile produces
when drawn (`drawImage` makes the summed pixel values equal `flux`; surface brightness is
`flux / arcsec²`). GalSim attaches a physical unit to flux *only* for chromatic objects built from a
`galsim.SED × galsim.Bandpass`; a plain `.withFlux(x)` or `flux=x` treats `x` as whatever convention
the caller imposes.

The codebase already relies on this convention-carried interpretation in two different ways:

| Site | Meaning of `flux` there |
| ---- | ----------------------- |
| `pupils.py:207` — `galsim.TopHat(r_i, flux=(r_i / r_o) ** 2)` | dimensionless *area* weight |
| `components.py:113` — `.withFlux(self.relative_flux())` | `reflectivity × area` in m² (relative) |

There is therefore **no unit conflict at the GalSim boundary** — GalSim imposes none. The dimensional
chain from the photometry layer is clean and lands in ADU:

```
photon_flux [ph/(s·m²)] · exptime [s] · qe [e⁻/ph] · area [m²] / gain [e⁻/adu]  =  adu
```

(`photon_flux_to_adu`, `conversions.py:26-50`). So `profile.withFlux(adu.to_value(u.adu))` is a
legitimate bridge: the drawn streak then integrates to that many ADU.

**Decision (confirmed with the maintainer, 2026-07-31):** the feature adopts the convention
**"metroid GalSim flux ≡ total ADU"** from `ThroughputCurve.calculate_adu(magnitude, photo_params)`.

### The one nuance: ADU vs electrons

ADU-vs-electrons only becomes physically distinguishable if **Poisson noise** is later added, because
shot noise is Poissonian in electrons/photons, not ADU (a strictly correct noisy sim would scale in
electrons, apply Poisson, then `/gain → ADU`). `metroid` currently renders a **noiseless**
surface-brightness profile, so flux = ADU is correct today. This is captured as a documented
assumption, not built for now.

## The bridge and why `withFlux` distributes for free

The core operation is:

```
magnitude ──ThroughputCurve.calculate_adu──► total ADU ──.withFlux──► scaled profile
```

For a **composite**, the maintainer's directive is: *scale the total flux over the composite, then
distribute the scaled flux across components using their relative scalings.* This is exactly what
`galsim`'s `withFlux` does when applied to the summed profile, with no per-component loop:

- `CompositeOrbitalObject.profile` is `_project(galsim.Sum(component_profiles))`.
- `galsim.Sum` preserves each summand's flux; `_project` (rotate / shear / rotate / `÷ mu`) scales the
  whole sum **uniformly**, so it preserves the *ratios* between component fluxes.
- `summed.withFlux(total_adu)` rescales the entire object so it integrates to `total_adu`, uniformly,
  which preserves those same ratios. Each component therefore ends up with

  ```
  flux_i = total_adu × (reflectivity_i · area_i) / Σ_j (reflectivity_j · area_j)
  ```

This resolves the assumption flagged in the composite plan: the arbitrary absolute scale of
`relative_flux` (m²) is **normalized away** by `withFlux`; only the relative weights survive, and the
absolute scale is set by the magnitude. No change to `Component.relative_flux` is required — it was
designed as the extension point for precisely this step.

### Where the flux must be applied: after convolution

For the *tracked* profile, the ADU flux must be applied to the **final convolved profile**, not the
bare `profile`, because the pupil defocus profile is **not** unit-flux:
`AnnularPupil.get_profile` returns `TopHat(r_o) − TopHat(r_i, flux=(r_i/r_o)²)`, whose total flux is
`1 − (r_i/r_o)²` (`pupils.py:207`). GalSim convolution multiplies summand fluxes, so scaling the bare
profile first and then convolving would corrupt the absolute normalization. Applying `withFlux` to
the `galsim.Convolve` result sets the final integrated flux to exactly `total_adu` regardless of
intermediate flux bookkeeping. The untracked `get_scaled_profile` (for studying a bare profile
without a PSF/pupil) scales `self.profile` directly.

## API shape

Per the maintainer's direction, the logic lives at **two layers**, with the observatory routing
through the object-level method so a user can also work with an `OrbitalObject` standalone (to study
profiles without constructing a full `Observatory`):

**`OrbitalObject` (object level — the bridge + scaling):**
- `calculate_flux(brightness_spec, throughput, photo_params) -> Adu[Scalar]` — the single, isolated
  magnitude→ADU bridge, delegating to `throughput.calculate_adu`.
- `get_scaled_profile(brightness_spec, throughput, photo_params) -> galsim.GSObject` — bare profile
  scaled so it integrates to the ADU total (for profile studies).
- `get_scaled_tracked_profile(brightness_spec, throughput, photo_params, psf, pupil) -> galsim.Convolution`
  — tracked profile with the ADU applied after convolution.

`CompositeOrbitalObject` inherits all three unchanged; distribution across components is automatic
(above). No override needed — only tests asserting the distribution invariant.

**`Observatory` (orchestration — convenience):**
- `get_scaled_profile(orbital_object, brightness_spec, band, exptime, psf=None)` — pulls the band's
  `ThroughputCurve` from `self.camera`, builds `PhotometricParameters` via `get_photo_params`, and
  routes to the object-level method (tracked when a `psf` is supplied, using `self.pupil`).

**Reversibility (issue #35: "should work both ways"):**
- `ThroughputCurve.calculate_ab_magnitude_from_adu(adu, photo_params) -> float` inverts the linear
  AB relation: `mag = −2.5 · log₁₀(adu / adu₀)` where `adu₀ = calculate_adu(0.0, photo_params)`.

## Scope decision: core vs. canonical workflow

Issue #35 proposes a four-step *canonical-flux* workflow (standardized magnitude at a canonical
height/zenith → scaling terms → canonical flux → scale tracked profile) and lists two "improvements":
**range** flux scaling (`∝ 1/distance²`) and **projection** flux scaling (relaxing the flux-conserving
`÷ mu` in `_project`).

The maintainer's directive for *this* task narrows the immediate deliverable to: **input is always a
magnitude → photometry → flux → scale the profile**, plus **composite distribution**. That is Layer A
below. The canonical/range/projection workflow (Layer B) is a genuine *superset* with distinct
physics, and one part of it (relaxing `_project` flux conservation) mutates a shared code path that
only just stabilized across the merged pointing (#42) and composite (#41) work.

| Layer | Content | Status in plan |
| ----- | ------- | -------------- |
| **A (core)** | magnitude→ADU bridge; scaled (tracked) profile; composite distribution; adu→mag reverse | Fully specified — Milestones 1–4. Satisfies the maintainer's directive. |
| **B (canonical)** | canonical magnitude at reference geometry; range scaling `(d_can/d)²`; projection flux relaxation; reversible canonical↔observed | Fully specified — Milestone 5. Implements the issue's proposed workflow; **gated on a decision** about relaxing `_project`'s flux conservation. |

The two layers unify: Layer A is Layer B with the scaling terms set to identity (canonical distance =
observed distance, projection flux-conserving). Building A first and B on top avoids reworking the
object-level API.

### Why range scaling needs the canonical concept (not in Layer A)

If the input magnitude is the **observed** magnitude at the actual geometry, the object's distance is
*already* baked into that brightness — multiplying by `1/d²` again would double-count. Range scaling
is meaningful only relative to a **canonical** reference distance, which is the entire point of the
canonical-magnitude concept. Hence range/projection scaling belongs in Layer B, not A. Layer A takes
the magnitude at face value as the observed brightness.

## Risks (summary; full table in the plan)

- **`profiles → photometry` import.** Verified no cycle exists (`photometry` does not import
  `profiles`), so the object-level method may import `ThroughputCurve`/`PhotometricParameters`.
- **Applying flux before convolution** would mis-normalize because defocus is non-unit-flux — mitigated
  by applying `withFlux` to the convolved result.
- **Relaxing `_project` flux conservation (Layer B)** changes behavior for *all* objects and
  contradicts the current README invariant ("`÷ mu` conserves total flux … `mu > 1` would spuriously
  amplify flux"). Gated behind an explicit decision and confined to Milestone 5.
