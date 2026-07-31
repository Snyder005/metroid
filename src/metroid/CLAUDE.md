# metroid/

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `observatory.py` | `Observatory` class composing a `Camera`, `Pupil`, and `EarthLocation`; `get_photo_params` method; `get_scaled_profile` (resolves bandpass + photometric params and routes to the object-level flux scaling, tracked when a `psf` is given); `from_config` (delegates nested camera/pupil/location blocks) and `from_standard(name)` (bundled catalogue) factories | Implementing or modifying top-level observatory construction; debugging `PhotometricParameters` creation, magnitude→flux profile scaling, or config-driven / standard-object construction |
| `camera.py` | `Camera` class holding named `ThroughputCurve` bandpasses, gain, pixel scale, and quantum efficiency; `from_config` factory building bandpasses from speclite filter names, a group wildcard, or inline arrays | Implementing or modifying camera construction; debugging bandpass lookup, iteration, or config-driven bandpass loading |
| `__init__.py` | Empty package init | Checking what the top-level package exports |

## Subdirectories

| Directory | What | When to read |
| --------- | ---- | ------------ |
| `photometry/` | Radiometry layer: `ThroughputCurve`, `Sed`, `PhotometricParameters`, flux/ADU conversion functions | Implementing photometric calculations; debugging flux or ADU outputs |
| `profiles/` | Telescope pupil geometry and orbital object surface-brightness profiles | Implementing or modifying pupil shapes, orbital object geometry, or tracked galsim profiles |
| `utils/` | Unit enforcement machinery: quantity specs, `@enforce_units` decorator, config validation; the declarative config layer (`config.py`: YAML loader, catalogue resolver, `Registrable` registry) | Adding a new physical quantity; debugging unit validation errors; modifying enforcement behavior; implementing config/standard-object construction |
| `data/` | Bundled package data: `standard_objects.yaml` catalogue resolved by `Observatory.from_standard` | Adding or editing a standard object definition (e.g. a new named observatory) |
