# Continuous Satellite Pointing Angle

## Overview

`OrbitalObject` orientation relative to the observer is currently a binary boolean,
`nadir_pointing`. When `True`, the object's flat face points straight down (nadir) and the
line-of-sight foreshortening in `_project()` is applied; when `False` ("observatory pointing"),
the face points directly at the telescope and `_project()` is skipped entirely
(`orbital_objects.py:278-282`, `335-339`). These are the two extremes of a physical continuum. This
plan replaces the boolean with a continuous `pointing_angle` quantity so any orientation between the
two extremes can be modeled, with **nadir-pointing as the default**.

The key insight (given in the task context) is that `_project()` should stop being toggled and
instead *always* run, parameterized by the pointing angle. The projection foreshortens by
`mu = cos(effective_angle)`, where the effective angle between the object's surface normal and the
line of sight varies continuously with pointing. At the nadir extreme the effective angle equals
`nadir_angle` (reproducing today's `mu = cos(nadir_angle)`); at the observatory extreme it is `0`
(reproducing today's no-op, `mu = cos(0) = 1`). Modeling `effective_angle = nadir_angle -
pointing_angle` (with `pointing_angle` measured from nadir toward the observatory line) gives a
single continuous expression bounded by, and reducing exactly to, the two current cases.

## Planning Context

This section is consumed VERBATIM by downstream agents (Technical Writer, Quality Reviewer).

### Decision Log

| Decision | Reasoning Chain |
| -------- | --------------- |
| Replace the boolean `nadir_pointing` with a continuous `pointing_angle: Angle[Scalar]`, default `0 deg` | The task requires arbitrary orientations, not two settings -> a boolean cannot express intermediate angles -> a continuous `Angle` quantity spans the range, and defaulting to `0 deg` (nadir) preserves the documented default behavior. |
| Measure `pointing_angle` from the nadir direction, toward the observatory line of sight | The two existing endpoints are "nadir" and "pointing at observatory"; the geometric difference between them is exactly `nadir_angle` (the angle at the object between its nadir and the telescope) -> measuring from nadir makes `pointing_angle = 0` the nadir case and `pointing_angle = nadir_angle` the observatory case -> the parameter's meaningful range is `[0, nadir_angle]` and the endpoints map cleanly onto today's two branches. |
| `_project()` is always applied; the foreshortening factor becomes `mu = cos(nadir_angle - pointing_angle)` | Today `_project` uses `mu = cos(nadir_angle)` and is skipped for observatory pointing (equivalent to `mu = 1`) -> substituting the effective angle `nadir_angle - pointing_angle` yields `cos(nadir_angle)` at `pointing_angle = 0` and `cos(0) = 1` at `pointing_angle = nadir_angle` -> one continuous expression reproduces both current branches and removes the conditional entirely. |
| Remove the `if self.nadir_pointing: ... else: ...` branch in every concrete `profile` and always return `self._project(profile)` | With `mu` continuous and equal to `1` at the observatory extreme, `_project` at that extreme is an identity-scale transform (rotate, transform by `(1,0,0,1)`, rotate back, divide by 1) -> the `else` branch (return unprojected) is now redundant -> collapsing to a single `return self._project(profile)` removes duplicated control flow across `CircularOrbitalObject` and `RectangularOrbitalObject`. |
| Keep `nadir_angle` and `rotation_angle` unchanged; `pointing_angle` is a new, independent axis | `nadir_angle` is a derived orbital geometry property (`orbital_objects.py:91-97`) and `rotation_angle` sets the in-plane azimuth used by `_project` (`orbital_objects.py:242`) -> pointing is a separate physical degree of freedom (tilt of the body normal within the observation plane) -> introducing it as its own parameter avoids overloading existing angles. |
| Validate `pointing_angle` and surface a clear error if outside the physically meaningful range | Values outside `[0, nadir_angle]` correspond to the object tilting past the observer or past nadir, which the two-endpoint model does not define -> silently accepting them would produce `mu > 1` (flux amplification) or `mu` for an un-modeled regime -> validate/clamp with an explicit message so misuse is caught, not hidden. (Exact policy — hard error vs. clamp — is a micro-decision for the Developer; default to raising `ValueError`.) |

### Rejected Alternatives

| Alternative | Why Rejected |
| ----------- | ------------ |
| Keep `nadir_pointing` boolean and add a *separate* `pointing_angle` used only when a third "custom" mode is selected | Reintroduces mode branching the task explicitly wants replaced by continuous application; two overlapping knobs (bool + angle) invite inconsistent states (e.g. `nadir_pointing=True` with a nonzero angle). |
| Interpolate `mu` linearly between `cos(nadir_angle)` and `1` as a function of a `[0,1]` blend factor | A unitless blend factor has no physical meaning and would not correspond to a real tilt angle; the cosine-of-effective-angle form is the actual projection geometry and is exact at both endpoints. |
| Deprecate `nadir_pointing` with a shim that maps `True`/`False` to angles | Adds long-lived compatibility surface for a pre-release library with few call sites; a clean parameter swap plus updating call sites/tests is simpler. Migration is noted in Known Risks instead. |
| Parameterize by the effective angle directly (user passes `nadir_angle - pointing_angle`) | Forces the user to compute a derived orbital quantity (`nadir_angle`) to express a simple physical intent ("the satellite is tilted X degrees off nadir"); pointing-from-nadir is the natural user-facing input. |

### Constraints & Assumptions

- **Existing behavior to preserve exactly**: at the default, results must be bit-for-bit equivalent
  to today's `nadir_pointing=True` (i.e. `mu = cos(nadir_angle)`); at `pointing_angle = nadir_angle`
  results must equal today's `nadir_pointing=False` (unprojected).
- **Unit machinery**: `pointing_angle` uses the existing `Angle` spec / `ANGLE` (`quantities.py:351`,
  `401`) with `Scalar` shape; `@enforce_units` on the constructor param and property, following the
  `rotation_angle` pattern (`orbital_objects.py:66-77`).
- **`_project` internals**: only the computation of `mu` changes (`orbital_objects.py:241`); the
  `rotate(phi).transform(mu,0,0,1).rotate(-phi) / mu` structure is retained
  (`orbital_objects.py:242-244`).
- **Affected call sites**: constructors of `OrbitalObject`, `CircularOrbitalObject`,
  `RectangularOrbitalObject` (`orbital_objects.py:27-38`, `250-259`, `296-306`); the two `profile`
  branches (`278-282`, `335-339`); and any tests/examples passing `nadir_pointing`.
- **Coordination**: the composite-satellite roadmap item (issue #41) also calls `_project` in its
  `profile`; both should end on the same single-`return self._project(...)` code path. If #41 lands
  first, update its call site too.
- **Dependencies**: `astropy.units`, `numpy`, `galsim` — already present.

### Known Risks

| Risk | Mitigation | Anchor |
| ---- | ---------- | ------ |
| Removing the `nadir_pointing` boolean is a breaking API change for any caller/test passing it. | Grep and update all constructor call sites and tests in the same change; note the parameter rename in the PR description. Pre-release library, few callers. | `profiles/orbital_objects.py:33` (param), `278`, `335` (uses). |
| Sign/direction convention of `pointing_angle` (measured from nadir vs. from observatory) could invert the mapping, breaking the endpoint equivalence. | Pin the convention in code + docstring + a test asserting `pointing_angle=0` reproduces old `nadir_pointing=True` and `pointing_angle=nadir_angle` reproduces old `nadir_pointing=False`. | `profiles/orbital_objects.py:241` (`mu = np.cos(self.nadir_angle)`). |
| `mu` could exceed 1 (unphysical flux gain via the `/ mu` term) if `pointing_angle` is outside `[0, nadir_angle]`. | Validate range in the setter and raise `ValueError` (default policy); document the meaningful range. `nadir_angle` depends on `zenith_angle`/`height`, so the valid upper bound is object-specific — validation must read `nadir_angle` at check time or defer to profile-build time. | `profiles/orbital_objects.py:244` (`/ mu`). |
| `nadir_angle` is `0` at zero zenith angle, collapsing the valid range to a single point `{0}`. | At `zenith_angle = 0` the object is directly overhead and pointing is degenerate; `pointing_angle = 0` is the only valid value and `mu = 1`. Confirm `distance`/`nadir_angle` degenerate branch (`orbital_objects.py:105-108`) stays consistent; add a test at `zenith_angle = 0`. | `profiles/orbital_objects.py:99-108` (`nadir_angle`, `distance` zero-zenith branch). |

## Invisible Knowledge

Technical Writer: update `src/metroid/profiles/README.md`.

1. **Business rule — pointing is a continuum, not a mode**: A satellite's orientation toward the
   observer is a continuous tilt. `pointing_angle` is measured from the object's nadir direction
   toward the telescope line of sight. `pointing_angle = 0` (the default) is nadir-pointing;
   `pointing_angle = nadir_angle` is "observatory-pointing" (face-on to the telescope). The
   physically meaningful range is `[0, nadir_angle]`, and `nadir_angle` itself depends on the
   object's `zenith_angle` and `height`.

2. **System invariant — projection is always applied**: `_project()` runs for every profile. The
   old boolean toggle is gone. The foreshortening factor `mu = cos(nadir_angle - pointing_angle)`
   equals `cos(nadir_angle)` at the nadir default and `1` (identity transform) at the observatory
   extreme, so the continuous form strictly generalizes the two former branches. This is why the
   per-shape `if nadir_pointing` conditional could be removed.

3. **Historical context**: Before this change, orientation was the boolean `nadir_pointing` with
   `_project` skipped entirely for observatory pointing. The continuous angle subsumes both former
   states; see the Decision Log for the endpoint-equivalence derivation.

4. **Invariant — `mu <= 1`**: The `/ mu` term in `_project` preserves total flux under
   foreshortening; `mu > 1` would spuriously amplify flux. This is why `pointing_angle` must stay
   within `[0, nadir_angle]` and is validated.

## Milestones

### Milestone 1: Introduce `pointing_angle` on `OrbitalObject`

**Files**: `src/metroid/profiles/orbital_objects.py`

**Code Intent**:

- Replace the `nadir_pointing: bool = False` constructor parameter with
  `pointing_angle: Angle[Scalar] = 0.0 * u.deg` on `OrbitalObject.__init__`
  (`orbital_objects.py:27-38`). Apply `@enforce_units`.
- Replace the `nadir_pointing` property/setter (`orbital_objects.py:79-89`) with a unit-enforced
  `pointing_angle` property/setter mirroring the `rotation_angle` pattern
  (`orbital_objects.py:66-77`). In the setter, validate the value lies in `[0, nadir_angle]` and
  raise `ValueError` with a clear message if not (Decision Log: validation; note `nadir_angle`
  depends on other state — see Known Risks for check-time considerations).
- Remove the `nadir_pointing` boolean entirely (Rejected Alternatives: no shim).

### Milestone 2: Make `_project` continuous

**Files**: `src/metroid/profiles/orbital_objects.py`

**Code Intent**:

- In `_project` (`orbital_objects.py:227-244`), change the foreshortening factor from
  `mu = np.cos(self.nadir_angle)` to `mu = np.cos(self.nadir_angle - self.pointing_angle)`. Keep the
  `rotate(phi).transform(mu, 0, 0, 1).rotate(-phi) / mu` structure unchanged.
- Verify the endpoint equivalence in reasoning: `pointing_angle = 0` → `mu = cos(nadir_angle)`
  (old nadir case); `pointing_angle = nadir_angle` → `mu = 1` → identity transform (old observatory
  case).

### Milestone 3: Collapse the per-shape `profile` branches

**Files**: `src/metroid/profiles/orbital_objects.py`

**Code Intent**:

- In `CircularOrbitalObject.profile` (`orbital_objects.py:270-282`) and
  `RectangularOrbitalObject.profile` (`orbital_objects.py:326-339`), remove the
  `if self.nadir_pointing / else` branch and always `return self._project(profile)`.
- Update both concrete constructors (`orbital_objects.py:250-259`, `296-306`) to accept and forward
  `pointing_angle` instead of `nadir_pointing` to `super().__init__`.
- Update docstrings referencing nadir/observatory pointing to describe the continuous angle.

### Milestone 4: Documentation

**Files**: `src/metroid/profiles/README.md`, `src/metroid/profiles/CLAUDE.md`

**Code Intent**:

- Technical Writer updates `profiles/README.md` with the Invisible Knowledge (continuum semantics,
  always-project invariant, endpoint equivalence, `mu <= 1`).
- Update the `orbital_objects.py` row in `profiles/CLAUDE.md` if its description references the
  pointing toggle.

### Milestone 5: Tests

**Files**: `tests/profiles/test_orbital_objects.py`

**Code Intent**:

- Replace any tests using `nadir_pointing`. Add:
  - **Endpoint equivalence**: at `pointing_angle = 0`, the profile matches the pre-change
    `nadir_pointing=True` behavior (projected with `mu = cos(nadir_angle)`); at
    `pointing_angle = nadir_angle`, it matches the old `nadir_pointing=False` unprojected profile
    (e.g. equal `mu`, equal centroid/second moments, or flux).
  - **Monotonic behavior**: an intermediate `pointing_angle` produces a `mu` strictly between the
    two endpoints.
  - **Validation**: `pointing_angle` outside `[0, nadir_angle]` raises `ValueError`; wrong
    unit/shape raises via `@enforce_units`.
  - **Degenerate geometry**: `zenith_angle = 0` (so `nadir_angle = 0`) accepts only
    `pointing_angle = 0` and yields `mu = 1` (Known Risks).
- Run `black`, `mypy`, `pytest` (project CLAUDE.md tooling).
