# profiles/

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `pupils.py` | `Pupil` ABC with `from_config` registry dispatch; `CircularPupil` and `AnnularPupil` concrete classes with `area` and `get_profile` returning galsim aperture profiles | Implementing or modifying telescope aperture shapes; debugging pupil construction from config; adding a new pupil type |
| `orbital_objects.py` | `OrbitalObject` ABC with orbital mechanics properties (distance, velocity, angular velocity, solid angle) and `get_tracked_profile`; `CircularOrbitalObject` and `RectangularOrbitalObject` concrete classes with galsim surface-brightness profiles | Implementing or modifying orbital object geometry; debugging profile construction or pixel traversal time; adding a new orbital object shape |
| `__init__.py` | Public exports: `Pupil`, `CircularPupil`, `AnnularPupil`, `OrbitalObject`, `CircularOrbitalObject`, `RectangularOrbitalObject` | Checking what `metroid.profiles` exposes |
