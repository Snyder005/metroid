# Flux Scaling — Implementation Report (Issue #35)

## Purpose

This report accompanies `plans/flux-scaling.md`. It records the design investigation behind the
flux-scaling feature: the physics, the GalSim/photometry unit reconciliation, the canonical-magnitude
model, the projection-flux decision, and the projection *seam* laid now for the future 3D-orientation
roadmap. The plan is the implementation directive; this report is the *why*. All open decisions have
been resolved with the maintainer (2026-07-31); they are recorded here and in the plan's Decision Log.

## Problem

`metroid` builds a satellite surface-brightness profile (`OrbitalObject.profile`) whose absolute scale
is arbitrary:

- `CircularOrbitalObject` / `RectangularOrbitalObject` return a `galsim.TopHat` / `galsim.Box` at unit
  flux, then apply a projection transform.
- `CompositeOrbitalObject` sums component profiles weighted by a *relative* `reflectivity × area`
  (`Component.relative_flux`, `profiles/components.py:67-84`). The composite plan states this is
  intentional: "absolute photometric normalization (magnitude → ADU) is intentionally deferred to the
  flux-scaling roadmap (issue #35)."

The observatory-facing brightness quantity is a **magnitude** (what telescopes report). Nothing
currently converts a magnitude into the absolute flux a rendered profile should integrate to. Issue
#35 closes that gap, in both directions (magnitude ↔ flux), and reconciles brightness across
observation geometries via a canonical reference.

## Investigation 1: what unit is "GalSim flux"?

A `galsim.GSObject.flux` is a **unitless scalar**: the total integrated counts the profile produces
when drawn. GalSim attaches a physical unit only for chromatic objects built from `SED × Bandpass`; a
plain `.withFlux(x)` treats `x` as whatever convention the caller imposes. The codebase already relies
on this: `pupils.py:207` uses flux as a dimensionless area weight; `components.py:113` uses
`reflectivity × area` (m²).

The photometry chain is dimensionally clean and lands in ADU:

```
photon_flux [ph/(s·m²)] · exptime [s] · qe [e⁻/ph] · area [m²] / gain [e⁻/adu]  =  adu
```

(`photon_flux_to_adu`, `conversions.py:26-50`). **Decision:** adopt the convention
**"metroid GalSim flux ≡ total ADU"** from `ThroughputCurve.calculate_adu(magnitude, photo_params)`.

### ADU vs electrons (the one nuance)

ADU-vs-electrons is only distinguishable if **Poisson noise** is later added, because shot noise is
Poissonian in electrons/photons, not ADU. `metroid` renders a **noiseless** profile, so flux = ADU is
correct today. Documented as an assumption, not built for now.

## Investigation 2: the canonical magnitude (500 km, zenith)

Issue #35 proposes standardizing brightness at "a canonical orbital height observed at Zenith." The
maintainer fixed the reference at **500 km height, zenith angle 0** — call it `d_can` (= 500 km, since
at zenith `distance == height`) with `mu_can = 1`.

The construction-time magnitude is interpreted as the **observed** AB magnitude at the object's actual
geometry. The conversion to the canonical magnitude is **purely geometric** — a flux ratio, so it is
band-independent and needs no photometry:

```
flux ∝ mu / distance²     (projection foreshortening × inverse-square range)

m_canonical = m_observed − 2.5·log₁₀(flux_can / flux_obs)
            = m_observed + 2.5·log₁₀(mu_obs) − 5·log₁₀(d_obs / d_can)
```

where `mu_obs = cos(nadir_angle − pointing_angle)` and `d_obs = self.distance`. The inverse (canonical
→ observed at *any* geometry) is:

```
m_observed(geom) = m_canonical − 2.5·log₁₀(mu(geom)) + 5·log₁₀(d(geom) / d_can)
```

This is exactly the maintainer's described flow: take the observed magnitude, "reverse through the
projection and radius scaling" to get the 500 km magnitude. It satisfies issue #35's "should work both
ways."

### Why store the *canonical* magnitude, not the observed one

`height`, `zenith_angle`, and `pointing_angle` are **mutable setters** on `OrbitalObject`
(`orbital_objects.py:52-105`). A stored *observed* magnitude would silently go stale the moment any of
those change, because the observed brightness depends on the geometry. The **canonical** magnitude is
geometry-invariant by definition, so it is the correct thing to store; the observed magnitude for the
current geometry is re-derived on demand. This is the robust choice against the existing mutability.

### Lumos-Sat cross-check

Lumos-Sat (github.com/Forrest-Fankhauser/lumos-sat, since the RTD API pages 404) confirms the physics:

- `lumos.conversions.intensity_to_ab_mag`: `ab_mag = −2.5·log₁₀(intensity · λ / (c · 3631e-26))` — the
  same linear AB relation metroid's `ThroughputCurve` already uses, so the reverse conversion is
  consistent.
- `lumos.calculator.get_intensity_satellite_frame`:
  `intensity = SUN_INTENSITY · Σ[area · BRDF · sun_norm · observer_norm] / dist²`. Two things confirm
  our model: (a) **explicit `1/dist²` range scaling**, and (b) **projection enters as**
  `observer_normalization = clip(normal · observer_dir, 0)` — the `cos` foreshortening (our `mu`)
  **multiplies flux**. Their BRDF is angle-dependent; per issue #35 scope we treat it as diffuse/constant.
- `lumos.geometry.Surface` (area, normal, brdf) overlaps future `OrbitalObject`/`Component` surface
  properties (shared area, per-surface normal). Per the maintainer, incorporating those is a **later
  roadmap item, out of #35 scope** — noted so the seam below anticipates it.

## Investigation 3: projection flux — relax `÷mu`

`OrbitalObject._project` currently ends in `÷ mu` (`orbital_objects.py:257-260`), which *conserves*
total flux under foreshortening. Verified against GalSim's real behavior:

```
galsim.TopHat(1.0).transform(mu,0,0,1).flux        == mu     (0.5 for mu=0.5)  # Jacobian scales flux
galsim.TopHat(1.0).transform(mu,0,0,1).flux / mu   == 1.0                       # current ÷mu restores it
proj.withFlux(100).flux                            == 100.0                     # withFlux overrides total
```

**Decision (maintainer):** **relax `÷mu`** — drop it, so `transform(mu,0,0,1)` leaves total flux
scaled by `mu` (projected-area dimming, matching Lumos' `observer_normalization`). Consequences:

- **Output-neutral for scaled profiles today.** `withFlux(observed_ADU)` overrides the total for single
  objects; for a shared-angle composite the common `mu` cancels in the `wᵢ/Σwⱼ` distribution. So this
  changes no scaled-profile result now.
- **Changes bare-`profile` flux** (1.0 → `mu`) and contradicts the current README invariant ("`÷mu`
  conserves total flux … `mu > 1` would spuriously amplify flux"). The README and any bare-profile
  flux test must be updated.
- **Payoff is future:** projected-area flux weighting only becomes observable when components have
  independent orientations (differing `muᵢ`) — the 3D roadmap. Relaxing now means that model needs no
  flux-semantics change later.

## Investigation 4: the projection seam (for future independent/3D component orientation)

Today `CompositeOrbitalObject.profile` projects the *summed* profile once
(`orbital_objects.py:394-405`). Because `_project` is a linear transform about the body-frame origin
and `galsim.Sum` is linear:

```
project(Σ shiftᵢ(Pᵢ))  ≡  Σ project(shiftᵢ(Pᵢ))
```

So relocating projection to a **per-component step** — `galsim.Sum([self._project(c.get_profile(d))
for c in components])` — is **provably output-identical** while the composite drives every component
with the same shared angle.

**Decision (maintainer): lay the seam now.** The composite still triggers projection with one shared
angle (identical results), but the *structure* is per-component. When the far-off roadmap adds
independent component orientation (or a full 3D body projected to 2D), the shared `self._project(...)`
call becomes component-aware — each component computing its own `mu` from its own body-frame normal +
the shared viewing geometry — **without another structural pass over `orbital_objects.py`**. This
pairs naturally with the `÷mu` relaxation: once `muᵢ` differ per component, an edge-on panel correctly
contributes less flux.

**When to revisit the seam (documented for the future):** promote the shared-angle
`self._project(component_profile)` to a per-component projection **when any of these becomes true** —
(1) components gain independent orientation/pointing (own normals), (2) a 3D body representation is
projected to a 2D observed profile, or (3) per-component projected-area flux weighting is required
(e.g. a deployed solar panel seen edge-on). Until then the shared angle keeps behavior identical to
project-once-on-the-sum.

## The core bridge and where flux is applied

```
observed magnitude ──(geometric)──► canonical magnitude   [stored, geometry-invariant]
canonical magnitude ──(geometric, current geom)──► observed magnitude
observed magnitude ──ThroughputCurve.calculate_adu──► total ADU ──withFlux──► scaled profile
```

For the **tracked** profile the ADU must scale the **convolved** output, not the bare profile, because
pupil defocus is not unit-flux: `AnnularPupil.get_profile` returns
`TopHat(r_o) − TopHat(r_i, flux=(r_i/r_o)²)`, total flux `1−(r_i/r_o)²` (`pupils.py:207`), and
convolution multiplies summand fluxes. `withFlux` divides by the current flux and multiplies by the
target, so applying it to the `galsim.Convolve` result normalizes to 1.0 and rescales to exactly the
ADU total regardless of intermediate bookkeeping — this is precisely the maintainer's "set the
convolution to 1.0 and then scale."

For a **composite**, `withFlux(total_adu)` on the summed/projected profile distributes the total across
components as `total_adu · wᵢ / Σⱼ wⱼ` (GalSim linearity), realizing "scale the total, distribute by
relative scalings" with no per-component loop. `Component.relative_flux` stays a *relative* weight; its
arbitrary m² scale is normalized away.

## API shape

Two entry points, one implementation (maintainer directive: allow studying a profile without a full
`Observatory`):

**`OrbitalObject` (object level):**
- Constructor gains optional `magnitude` (observed AB mag at construction geometry; default `None` =
  no absolute scaling = current behavior). Converted immediately to and stored as `_canonical_magnitude`.
- `canonical_magnitude -> float | None` property; `observed_magnitude -> float | None` property
  (re-derived for current geometry).
- `calculate_flux(throughput, photo_params, brightness_spec=None) -> Adu[Scalar]` — the isolated
  magnitude→ADU bridge (uses `observed_magnitude` when `brightness_spec` is omitted).
- `get_scaled_profile(throughput, photo_params, brightness_spec=None) -> galsim.GSObject`.
- `get_scaled_tracked_profile(throughput, photo_params, psf, telescope_pupil, brightness_spec=None) -> galsim.Convolution`.

**`Observatory` (orchestration):**
- `get_scaled_profile(orbital_object, band, exptime, brightness_spec=None, psf=None)` — resolves
  `throughput = camera[band]` + `photo_params = get_photo_params(exptime)`, routes to the object-level
  method (tracked when `psf` supplied, with `self.pupil`).

**`ThroughputCurve` (reverse conversion):**
- `calculate_ab_magnitude_from_adu(adu, photo_params) -> float`: `mag = −2.5·log₁₀(adu/adu₀)` with
  `adu₀ = calculate_adu(0.0, photo_params)`.

## Scope summary

| Layer | Content | Status |
| ----- | ------- | ------ |
| Reverse conversion | `ThroughputCurve` ADU→magnitude | Milestone 1 |
| Canonical magnitude | observed↔canonical (500 km, zenith); stored canonical | Milestone 2 |
| Object bridge + scaling | `calculate_flux`, `get_scaled_profile`, `get_scaled_tracked_profile`; scale after convolution | Milestone 3 |
| Projection relaxation + seam | drop `÷mu` in `_project`; per-component projection in composite (shared angle now) | Milestone 4 |
| Composite distribution | tests + docs (no new production code) | Milestone 5 |
| Observatory orchestration | convenience delegation | Milestone 6 |

All six milestones are in scope for issue #35 given the maintainer's decisions; the `lumos.Surface`
per-surface normal/BRDF material and true independent/3D component orientation remain explicitly
**future roadmap**, with the seam (Milestone 4) laid to receive them.
