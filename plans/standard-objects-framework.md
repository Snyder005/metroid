# Framework for Standard Objects

## Overview

The fundamental instrument objects — `Observatory`, `Camera`, `Pupil` — must currently be
constructed by hand from arbitrary user-supplied quantities. This plan introduces a declarative
framework: standard objects are defined in a **YAML** file and instantiated through factory methods
keyed by an object **name/label**. A user can then write `Observatory.from_standard("rubin")` (or
load from an explicit path) instead of wiring up cameras, pupils, gains, and pixel scales by hand.

The design generalizes the pattern `Pupil` already uses — a subclass registry keyed by a `type`
string plus a `from_config(dict)` dispatcher (`pupils.py:15-112`) — into a small, shared
config-loading layer under `metroid.utils`, and adds `from_config`/`_from_config` support to
`Camera` and `Observatory`. A YAML document holds one or more named object definitions; the factory
looks a name up, reads its `type` (for polymorphic objects like `Pupil`), and delegates to the
appropriate class. Bandpass definitions lean on speclite: a `Camera` config may name individual
filters (`ThroughputCurve.load_filter`) **or** a whole filter group via wildcard
(`speclite.filters.load_filters("lsst2023-*")`), wrapping each returned `FilterResponse` in a
`ThroughputCurve`.

Scope is deliberately limited to the observatory-side stack (`Observatory`, `Camera`, `Pupil`).
Orbital objects/satellites are **not** covered here, but the framework is designed so the same
registry + `from_config` pattern extends to `OrbitalObject` subclasses later (see Invisible
Knowledge); that extension will interact with the composite-satellite roadmap item (issue #41).

## Planning Context

This section is consumed VERBATIM by downstream agents (Technical Writer, Quality Reviewer).

### Decision Log

| Decision | Reasoning Chain |
| -------- | --------------- |
| Use YAML as the config format | Standard-object definitions are human-authored and benefit from comments and nested structure -> JSON forbids comments and is noisier for nested quantities -> YAML is the common astronomy-config choice and keeps definitions readable; the parsed result is a plain `dict`, so the existing dict-based `from_config` machinery is reused unchanged. |
| Parse YAML into a `dict` and keep all `from_config` logic dict-based (format-agnostic core) | `Pupil.from_config` already operates on a `dict` (`pupils.py:27`) -> if the loader only turns YAML text into a dict and every class consumes dicts, the file format is isolated to one function -> a future JSON/TOML loader can be added without touching any class, and unit tests can pass dicts directly without file I/O. |
| Generalize the `Pupil` registry pattern into a reusable mixin/helper in `metroid.utils` rather than copy it onto `Camera`/`Observatory` | `Pupil` has a working `__init_subclass__` registry + `from_config` type-dispatch (`pupils.py:18-68`) -> `Camera`/`Observatory` need the same "look up name, read type, delegate" behavior -> extracting it once avoids three near-identical copies and keeps the dispatch semantics consistent; `Pupil` is refactored to consume the shared helper so there is a single implementation. |
| Two-level config: a top-level file maps a **name/label** -> an object definition; the definition carries a `type` for polymorphic classes | The task requires instantiation "based on an object name/label" (e.g. "rubin"), which is distinct from the polymorphic `type` (e.g. "annular") `Pupil` already uses -> separating the human label (which object) from the structural type (which subclass) lets one file hold many named standard objects, each selecting its subclass -> `Observatory.from_standard(name)` reads the named block, whose nested `camera`/`pupil` blocks each carry their own fields. |
| `Camera.from_config` supports bandpasses specified as (a) individual speclite filter names, (b) a speclite group wildcard (`load_filters("lsst2023-*")`), or (c) inline wavelength/throughput arrays | `Camera` holds a `dict[str, ThroughputCurve]` (`camera.py:16-34`); speclite already loads single filters (`throughput.py:67-90`) and, via `load_filters(...)`, whole camera groups returning a `FilterSequence` of `FilterResponse` -> wiring the group loader in lets a config say "all LSST 2023 bands" in one line instead of enumerating six -> each `FilterResponse` wraps via the existing `ThroughputCurve.from_filter_response`, so no new photometry code is needed. |
| Wildcard/group bandpass keys use the speclite band name (e.g. `"lsst2023-r"` or its short band `"r"`) derived from `FilterSequence.names` | `load_filters("lsst2023-*").names` yields fully-qualified names like `lsst2023-r` -> the `Camera` bandpass dict is keyed by `str`, and the natural, collision-free key is the speclite name (or its trailing band letter) -> deriving keys from `.names` avoids the user hand-listing keys and keeps camera lookup (`camera.py:36-40`) working with familiar band labels. (Exact key form — full name vs. band letter — is a micro-decision for the Developer; default to the band letter with the full name as fallback.) |
| `Observatory.from_config` composes by delegating nested blocks to `Camera.from_config`, `Pupil.from_config`, and building `EarthLocation` from a `location` block | `Observatory.__init__` already requires a `Camera`, a `Pupil`, and an `EarthLocation` (`observatory.py:13-27`) -> a config that nests `camera:`, `pupil:`, `location:` sub-dicts maps one-to-one onto those constructor args -> delegation reuses each component's own `from_config` and keeps `Observatory` ignorant of camera/pupil internals. `EarthLocation` is built from lat/lon/height (astropy) or a site name via `EarthLocation.of_site`. |
| Ship a small bundled catalogue of standard definitions (e.g. a `standard_objects.yaml` inside the package) resolved by `from_standard(name)`; also allow an explicit path | The task's "standard objects" implies named, reusable presets (e.g. Rubin) -> bundling a data file and resolving names against it gives `from_standard("rubin")` out of the box -> also accepting a path (`from_config`/`from_file`) keeps it open for user-supplied catalogues without editing the package. |

### Rejected Alternatives

| Alternative | Why Rejected |
| ----------- | ------------ |
| JSON as the primary format | No comments, poorer readability for nested physical quantities; less conventional for astronomy instrument configs. Kept feasible via the dict-based core, but not the default. |
| Put `from_config` logic directly in each class, copying the `Pupil` registry three times | Triplicates the registry/dispatch code and lets the three copies drift; a shared helper in `utils` is one implementation with consistent error messages. |
| A single flat config (no name→definition layer) | Cannot hold multiple named standard objects in one catalogue; the task explicitly wants selection "based on an object name/label", which needs the extra level. |
| Encode physical units as structured `{value, unit}` objects in YAML | Verbose and repetitive; instead adopt a simple convention (bare numbers in each field's documented canonical unit, converted on load via the existing quantity specs), matching how `Pupil._from_config` already reads bare floats and multiplies by `u.m` (`pupils.py:146-147`). |
| Include orbital objects/satellites in this framework now | Out of stated scope; their construction couples to the composite-satellite design (issue #41). The registry pattern is built to extend to them later without rework. |
| Add a heavy schema/validation library (e.g. pydantic) | Introduces a new dependency for a small config surface; the existing `get_field_value` typed extraction (`validation.py:4-40`) plus `@enforce_units` already give field-presence, type, and unit validation. |

### Constraints & Assumptions

- **Reuse existing validation**: field extraction via `get_field_value(config, name, dtype)`
  (`validation.py:4-40`); unit/shape validation via `@enforce_units` and the quantity specs
  (`quantities.py`). Follow `Pupil._from_config`'s "read bare float, attach canonical unit" idiom
  (`pupils.py:146-147`).
- **Registry pattern to generalize**: `Pupil.__init_subclass__` + `_registry` +
  `from_config`/`_from_config` (`pupils.py:18-87`). The refactor must preserve `Pupil`'s existing
  public behavior and error messages (`pupils.py:57-68`) — covered by existing
  `tests/profiles/test_pupils.py`.
- **speclite APIs**: `load_filter(name)` (single, already wrapped at `throughput.py:67-90`) and
  `load_filters("<group>-*")` → `FilterSequence` with `.names` and iterable `FilterResponse`
  elements; wrap each with `ThroughputCurve.from_filter_response` (`throughput.py:41-65`).
- **New dependency**: a YAML parser (`pyyaml`). Add to project dependencies/packaging. Use
  `yaml.safe_load`.
- **Package data**: a bundled `standard_objects.yaml` must be included as package data in
  the build/packaging configuration so `from_standard` resolves after install.
- **`EarthLocation`**: build from a `location` block (lat/lon/height) or `of_site(name)`; keep the
  isinstance guard in `Observatory.__init__` (`observatory.py:24-27`) intact.
- **Layout**: shared config helper under `src/metroid/utils/`; per-class `from_config`/`_from_config`
  on `Camera` (`camera.py`) and `Observatory` (`observatory.py`); `Pupil` refactored in place. Tests
  mirror under `tests/`.

### Known Risks

| Risk | Mitigation | Anchor |
| ---- | ---------- | ------ |
| Refactoring `Pupil`'s registry into a shared helper could change its public `from_config` behavior or error messages. | Keep `Pupil.from_config`'s signature and the two `ValueError` messages ("missing required field 'type'", "unknown pupil type: ...") identical; rely on existing `tests/profiles/test_pupils.py` to catch regressions and add tests for the shared helper. | `profiles/pupils.py:57-68`. |
| Adding a `pyyaml` dependency affects packaging and any minimal install. | Declare the dependency explicitly in packaging metadata; isolate `import yaml` to the loader module so the rest of the library imports without it if unused. | — (packaging/config; no code anchor). |
| Bundled `standard_objects.yaml` may not be installed as package data, breaking `from_standard` after `pip install`. | Register it as package data / include in the wheel; resolve via `importlib.resources` rather than a filesystem-relative path so it works from an installed package. | — (packaging). |
| Bandpass key collisions when mixing group wildcards with individually named filters, or short-band vs. full-name keys. | Define one key convention (default: speclite band letter, full name as fallback) derived from `FilterSequence.names`; raise a clear error on duplicate keys instead of silently overwriting. | `camera.py:27-34` (dict build already type-checks keys/values). |
| `EarthLocation.of_site` performs network/remote-catalogue access and can fail offline. | Prefer explicit lat/lon/height in configs for reproducibility; treat `of_site` as an optional convenience and document the offline caveat. | `observatory.py:24-27`. |
| Unit convention ambiguity: bare numbers in YAML rely on documented canonical units per field. | Document each field's expected unit in the class `_from_config` docstring (as `Pupil` does) and convert on load via the quantity specs; `@enforce_units` rejects mismatches downstream. | `profiles/pupils.py:122-147` (radius read as float → `* u.m`). |

## Invisible Knowledge

Technical Writer: create `src/metroid/utils/README.md` (config layer) and update
`src/metroid/README.md` / relevant package READMEs for the object factories.

1. **Architectural decision — format-isolated, dict-based core**: The only code that knows the file
   is YAML is the loader (`yaml.safe_load` → `dict`). Every `from_config` consumes a plain `dict`,
   so the format can change (JSON/TOML) by adding a loader, and classes/tests can bypass files
   entirely by passing dicts. This mirrors and generalizes `Pupil.from_config`.

2. **Architectural decision — name/label vs. type**: Two distinct selectors exist. The **name/label**
   (e.g. "rubin") picks *which* standard object from a catalogue; the **type** (e.g. "annular")
   picks *which subclass* for polymorphic objects. `from_standard(name)` resolves the label against
   the bundled catalogue; `from_config(dict)` resolves the type against the class registry.

3. **Extension point — registry generalizes to orbital objects**: The shared registry +
   `from_config` helper is intentionally class-agnostic. `OrbitalObject` subclasses (including the
   composite satellites of issue #41) can opt in by registering a `type` and implementing
   `_from_config`, so a future release can define satellites declaratively with no new machinery.
   This was deferred here only to keep scope tight and avoid coupling to the in-flight composite
   design.

4. **Business rule — bandpasses from speclite groups**: A camera's filter set can be declared as a
   speclite group wildcard (`lsst2023-*`). `load_filters` returns a `FilterSequence`; each member is
   a `FilterResponse` wrapped via `ThroughputCurve.from_filter_response`, keyed by its band name.
   This lets a standard camera name its entire filter complement in one line.

5. **Convention — bare numbers in canonical units**: Config fields carry bare numeric values in each
   field's documented canonical unit (e.g. radius in meters), converted on load and then validated
   by `@enforce_units`. There is no `{value, unit}` wrapper; the documented unit per field is the
   contract.

## Milestones

### Milestone 1: Shared config-loading and registry helper

**Files**: `src/metroid/utils/config.py` (new), `src/metroid/utils/__init__.py`

**Code Intent**:

- Add a YAML loader: `load_yaml(path) -> dict` using `yaml.safe_load`; and a resolver for bundled
  catalogues using `importlib.resources` (e.g. `load_standard_catalogue() -> dict`).
- Add a reusable registry mechanism generalizing `Pupil`'s: a `ConfigurableMixin` (or a small set of
  helpers) providing `__init_subclass__(type=...)` registration into a per-base `_registry`, and a
  `from_config(cls, dict)` classmethod that pops `type`, looks up the subclass, and delegates to the
  subclass `_from_config(dict)`. Preserve the exact error semantics used by `Pupil`
  (missing `type` → `ValueError`; unknown type → `ValueError`).
- Reuse `get_field_value` (`validation.py`) for typed field extraction inside the helper where
  useful.

### Milestone 2: Refactor `Pupil` onto the shared helper

**Files**: `src/metroid/profiles/pupils.py`

**Code Intent**:

- Replace `Pupil`'s bespoke `_registry` / `__init_subclass__` / `from_config` (`pupils.py:18-68`)
  with the shared mechanism from Milestone 1, keeping the public `from_config` signature and error
  messages identical.
- Leave `CircularPupil._from_config` / `AnnularPupil._from_config` behavior unchanged
  (`pupils.py:122-147`, `201-223`). Existing `tests/profiles/test_pupils.py` must still pass.

### Milestone 3: `Camera.from_config`

**Files**: `src/metroid/camera.py`

**Code Intent**:

- Add `Camera.from_config(cls, config: dict) -> Camera` that builds:
  - `bandpasses`: support three forms — a list/dict of individual speclite filter names
    (`ThroughputCurve.load_filter`), a group wildcard string (`load_filters("<group>-*")` → wrap each
    `FilterResponse` via `ThroughputCurve.from_filter_response`, keyed from `FilterSequence.names`),
    and inline `wavelength`/`throughput` arrays (`ThroughputCurve(...)`). Raise a clear error on
    duplicate bandpass keys.
  - `gain`, `pixel_scale`, `qe`: bare floats read via `get_field_value` and multiplied by their
    canonical units (mirroring `Pupil._from_config`), then passed to `Camera.__init__` where
    `@enforce_units` validates them (`camera.py:15-34`).
- Document each field's expected unit in the `from_config` docstring.

### Milestone 4: `Observatory.from_config` and `from_standard`

**Files**: `src/metroid/observatory.py`

**Code Intent**:

- Add `Observatory.from_config(cls, config: dict) -> Observatory` that delegates nested blocks:
  `camera:` → `Camera.from_config`, `pupil:` → `Pupil.from_config`, `location:` → `EarthLocation`
  (lat/lon/height, or `of_site(name)` as optional convenience).
- Add `Observatory.from_standard(cls, name: str) -> Observatory` that loads the bundled catalogue
  (Milestone 1), selects the block for `name` (clear `ValueError` if the label is unknown), and
  passes it to `from_config`. Optionally add `from_file(path, name=None)` for user catalogues.
- Keep the existing isinstance guards in `__init__` (`observatory.py:14-27`).

### Milestone 5: Bundled catalogue and packaging

**Files**: `src/metroid/data/standard_objects.yaml` (new), packaging config (e.g.
`pyproject.toml`/`setup.cfg`)

**Code Intent**:

- Author a starter `standard_objects.yaml` with at least one named object (e.g. `rubin`) exercising
  a group-wildcard camera, an annular pupil, and an explicit lat/lon/height location.
- Register the YAML as package data so it ships in the wheel; declare the `pyyaml` dependency.
- Resolve the file via `importlib.resources`, not a relative filesystem path.

### Milestone 6: Documentation

**Files**: `src/metroid/utils/README.md` (new), `src/metroid/utils/CLAUDE.md`,
`src/metroid/CLAUDE.md`, `src/metroid/profiles/CLAUDE.md`

**Code Intent**:

- Technical Writer creates `utils/README.md` capturing the Invisible Knowledge (format isolation,
  name-vs-type, registry extension point, unit convention).
- Update `utils/CLAUDE.md` with a `config.py` row; update `metroid/CLAUDE.md` (observatory/camera
  factory methods) and note the shared registry in `profiles/CLAUDE.md` (pupils refactor).

### Milestone 7: Tests

**Files**: `tests/utils/test_config.py` (new), `tests/test_camera.py`,
`tests/test_observatory.py`, `tests/profiles/test_pupils.py` (regression)

**Code Intent**:

- Config helper: `load_yaml` parses to dict; registry `from_config` dispatches on `type`, raises on
  missing/unknown type (assert identical messages to old `Pupil` behavior).
- `Camera.from_config`: builds bandpasses from individual names, from a group wildcard (assert all
  group bands present and correctly keyed), and from inline arrays; duplicate keys raise; gain/
  pixel_scale/qe carry correct units.
- `Observatory.from_config` / `from_standard`: composes camera + pupil + location; unknown standard
  name raises `ValueError`; a full round-trip from the bundled `standard_objects.yaml` yields a
  valid `Observatory` whose `get_photo_params` works (`observatory.py:46-70`).
- `Pupil` regression: existing tests still pass after the registry refactor.
- Where speclite group loading needs network/data access, mock or use a locally available speclite
  group to keep tests hermetic.
- Run `black`, `mypy`, `pytest` (project CLAUDE.md tooling).
