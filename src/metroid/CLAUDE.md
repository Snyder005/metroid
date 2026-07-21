# metroid/

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `observatory.py` | `Observatory` class composing a `Camera`, `Pupil`, and `EarthLocation`; `get_photo_params` method | Implementing or modifying top-level observatory construction; debugging `PhotometricParameters` creation |
| `camera.py` | `Camera` class holding named `ThroughputCurve` bandpasses, gain, pixel scale, and quantum efficiency | Implementing or modifying camera construction; debugging bandpass lookup or iteration |
| `__init__.py` | Empty package init | Checking what the top-level package exports |

## Subdirectories

| Directory | What | When to read |
| --------- | ---- | ------------ |
| `photometry/` | Radiometry layer: `ThroughputCurve`, `Sed`, `PhotometricParameters`, flux/ADU conversion functions | Implementing photometric calculations; debugging flux or ADU outputs |
| `profiles/` | Telescope pupil geometry and orbital object surface-brightness profiles | Implementing or modifying pupil shapes, orbital object geometry, or tracked galsim profiles |
| `utils/` | Unit enforcement machinery: quantity specs, `@enforce_units` decorator, config validation | Adding a new physical quantity; debugging unit validation errors; modifying enforcement behavior |
