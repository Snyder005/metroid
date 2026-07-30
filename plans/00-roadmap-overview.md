# Roadmap Overview: Integration Plan for Standard-Objects, Pointing, and Composites

This document sits above the three per-task plans and directs how they fit together. It is a
*coordination* plan, not a fourth feature: it does not introduce new code intent, only a recommended
order, the dependencies and overlaps between the tasks, and the context they share.

| Task | Issue | Branch | Plan |
| ---- | ----- | ------ | ---- |
| Continuous Satellite Pointing Angle | #42 | `feature/issue-42-continuous-pointing-angle` | `continuous-pointing-angle.md` |
| Composite Satellites from Components | #41 | `feature/issue-41-composite-satellites` | `composite-satellites.md` |
| Framework for Standard Objects | #43 | `feature/issue-43-standard-objects-framework` | `standard-objects-framework.md` |

## Recommended Order of Implementation

**1. #42 — Continuous Pointing Angle (do first).**
It is the smallest, most self-contained change and it *stabilizes the shared code path* the other
work depends on. It rewrites `OrbitalObject._project()` and collapses the per-shape
`if self.nadir_pointing: ... else: ...` branches into a single unconditional
`return self._project(profile)` in every concrete `profile`
(`orbital_objects.py:278-282`, `335-339`). Landing this first means the "project once, on the
profile" contract is fixed before anything else touches `orbital_objects.py`.

**2. #41 — Composite Satellites (do second).**
It adds `CompositeOrbitalObject`, whose `profile` calls `_project` on the summed component profile.
If #42 is already merged, #41 simply matches the new single-`return self._project(...)` pattern and
inherits continuous pointing for free — no reconciliation, no double-projection risk. If #41 landed
first instead, its `profile` would encode the *old* `if nadir_pointing` toggle and #42 would then
have to edit a third call site. Order 2-after-1 avoids that rework.

**3. #43 — Standard Objects Framework (do third, but largely parallelizable).**
Its core (YAML loader + shared registry helper, `Pupil` refactor, `Camera`/`Observatory`
`from_config`) is **orthogonal** to `orbital_objects.py` and can proceed in parallel with #42/#41.
It is sequenced last only because its *natural extension* — declaring satellites (including
composites) from config — depends on the composite type existing (#41). Do the observatory-side
framework whenever convenient; defer the orbital-object extension until after #41.

```
#42 (pointing)  ──►  #41 (composite)  ──►  [future: declare composites in config]
                                                    ▲
#43 (framework core, parallel) ─────────────────────┘  (extension point only)
```

## Dependencies and Overlaps

### Hard overlap: `orbital_objects.py` `_project` call site (#42 ↔ #41)

Both plans call `OrbitalObject._project()` from a `profile` property. This is the single most
important integration point.

- **#42** removes the toggle and makes every `profile` end in `return self._project(profile)`, with
  `_project` computing `mu = cos(nadir_angle - pointing_angle)`.
- **#41**'s `CompositeOrbitalObject.profile` builds `galsim.Sum(...)` then applies the same
  `_project`.

**Directive:** land #42 first so #41 is written against the final code path. Both per-task plans
already flag this (see `continuous-pointing-angle.md` → Constraints & Assumptions "Coordination";
`composite-satellites.md` → Milestone 2 note on issue #42). If they are developed concurrently on
separate branches, expect a merge conflict in the `profile` methods and resolve it toward the
single unconditional `_project` form.

### Soft dependency: registry pattern (#43 ↔ #41)

- **#43** generalizes the existing `Pupil` registry (`__init_subclass__` + `_registry` +
  `from_config`/`_from_config`, `pupils.py:18-87`) into a shared helper in `metroid.utils`, and is
  explicitly designed so `OrbitalObject` subclasses can later opt in.
- **#41** adds new `OrbitalObject` subclasses (`CompositeOrbitalObject`) and `Component` types.

These do not collide at the code level (different files), but the *future* "satellites from config"
capability is the product of both: #41 supplies the classes, #43 supplies the machinery. Neither
task needs to change for the other to merge; the join is a later, additive step. Keep #41's new
classes constructor-shaped so a `_from_config` can be bolted on without refactoring.

### No dependency

- **#43 core** (Pupil refactor, Camera/Observatory `from_config`, YAML loader, packaging) shares no
  files with #42 and no files with #41's `orbital_objects.py` changes. It can be built and merged at
  any time.
- **#42 ↔ #43** and **#41 ↔ #43 core** are independent.

## Shared Context Across All Three Tasks

These conventions recur in every plan; establishing them once avoids divergence.

1. **Unit enforcement is universal.** Every quantity-typed constructor param and property uses
   `@enforce_units` with a spec + shape marker from `metroid.utils.quantities`
   (`decorators.py:10`, `quantities.py`). New quantities (e.g. `pointing_angle` reuses `Angle`;
   composite reflectivity reuses/extends `Fraction`) follow this without exception.

2. **Physical-length → angular-size conversion has one idiom.**
   `(length / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())`
   (`orbital_objects.py:275`, `pupils.py:186`). #41's `Component.get_profile` and #43's config-built
   objects must use it; #42 does not add conversions but relies on the same `distance`/`nadir_angle`
   geometry (`orbital_objects.py:99-108`).

3. **GalSim is the manipulation engine.** #41 uses `TopHat`/`Box`/`shift`/`withFlux`/`Sum`; #42
   keeps the `rotate/transform/rotate` structure of `_project`. No hand-rolled pixel math.

4. **Read-only `_name`-backed properties + explicit isinstance guards** are the class idiom
   (`orbital_objects.py`, `observatory.py:14-27`, `camera.py:27-34`). New classes/params in all
   three tasks follow it, including the type/value error message style.

5. **Flux scaling stays relative; absolute normalization is out of scope (issue #35).** #41 stops at
   reflectivity×area *relative* brightness and explicitly defers absolute magnitude→ADU to #35. #42
   preserves the flux-conserving `/ mu` term. Neither task should introduce an absolute-flux path
   that would fork the #35 workflow. This is the one shared constraint most likely to cause scope
   creep — hold the line in all three.

6. **Documentation locality.** Each task updates the `CLAUDE.md` navigation table in the directory
   it touches and hands Invisible Knowledge to the Technical Writer for a co-located `README.md`
   (`profiles/` for #41/#42, `utils/` for #43). Keep these edits within the owning branch.

## Suggested Merge Sequence

1. Merge **#42** → `main`. Confirm `orbital_objects.py` tests green with the continuous angle.
2. Rebase **#41** on updated `main`; its `profile` should already match the single-`_project` path.
   Merge #41.
3. Merge **#43 core** any time (independent). 
4. *(Later, new issue)* Add `_from_config`/registry opt-in to `OrbitalObject`/`CompositeOrbitalObject`
   to declare satellites from config — the join of #41 and #43.

Steps 1–2 are strictly ordered by the shared `_project` call site. Step 3 floats. Step 4 is future
work gated on both #41 and #43.
