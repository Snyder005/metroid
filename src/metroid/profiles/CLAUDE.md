# profiles/

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `pupils.py` | `Pupil` ABC with `from_config` registry dispatch; `CircularPupil` and `AnnularPupil` concrete classes with `area` and `get_profile` returning galsim aperture profiles | Implementing or modifying telescope aperture shapes; debugging pupil construction from config; adding a new pupil type |
| `orbital_objects.py` | `OrbitalObject` ABC with orbital mechanics properties (distance, velocity, angular velocity, solid angle), a continuous `pointing_angle` and an always-applied `_project` foreshortening (`mu = cos(nadir_angle - pointing_angle)`), and `get_tracked_profile`; `CircularOrbitalObject` and `RectangularOrbitalObject` primitive shapes; `CompositeOrbitalObject` assembling `Component` parts into a summed, once-projected profile | Implementing or modifying orbital object geometry; debugging profile construction, pointing-angle projection, composite assembly, or pixel traversal time; adding a new orbital object shape |
| `components.py` | `Component` ABC (body-frame `x0`/`y0` offset, `reflectivity`, `relative_flux = reflectivity * area`, `get_profile(distance)` returning an unprojected shifted galsim profile); `CircularComponent` and `RectangularComponent` concrete parts | Implementing or modifying satellite component geometry; debugging per-component flux scaling or body-frame offsets; adding a new component shape |
| `__init__.py` | Public exports: `Pupil`, `CircularPupil`, `AnnularPupil`, `OrbitalObject`, `CircularOrbitalObject`, `RectangularOrbitalObject`, `CompositeOrbitalObject`, `Component`, `CircularComponent`, `RectangularComponent` | Checking what `metroid.profiles` exposes |
