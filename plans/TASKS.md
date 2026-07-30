# Tasks

The following sections each correspond to a task for which to develop a plan for. Each plan must be saved in a format following the outline provided in plan-format.md. These are not ordered in terms of implementation steps (for each create a plan in isolation), although some features may change the eventual implementation; this is problem for integration, not creating a basic plan. Each plan should be comprehensive enough for a developer to implement a specific code functionality without needed to rederive context. **Clear directive forward for implementation, but do not implement the code**.

## Composite Satellites Built from Components

Currently satellites are singular geometric objects (disc, rectangle). In reality a satellite is created from multiple components. The directive is to develop a framework to build a more complex satellite from combinations of simple shapes.

Additional context:
- Each component have a hook to scale relative to the other components in the satellite, i.e. one shape is brighter than the other. In reality these should be scaled by the reflectivity, which sets the relative surface brightness, take this into consideration.
- The underlying manipulation of surface brightness profiles is GalSim. This software provides transformations to position profiles relative to each other, combine into a single profile, and in principle, feed this complicated shape through the existing convolution tools. Priority should be to continue to use GalSim to drive this task.
- For reference an old implementation of similar is in 
  - https://github.com/Snyder005/leosim/blob/main/python/leosim/component.py
  - https://github.com/Snyder005/leosim/blob/main/python/leosim/orbital_object.py (CompositeOrbitalObject)
  Use these for guidance for intent, not a guidance for the actual code implementation, which may be different.

## Continuous Satellite Pointing Angle

Currently there are only two settings for the orientation of the satellite relative to the observer, set by the boolean OrbitalObject.nadir_pointing.  If `True` the satellite is nadir-pointing (straight down), if `False` the satellite points directly at the observatory. The directive is to expand this to allow arbitrary angles (although the default should be nadir-pointing). 

Important context:
- The satellite pointing, set by `OrbitalObject.nadir_point`, determines the application of projection effects through `OrbitalObject._project()`. Therefore that private method assumes nadir pointing, and if it is "observatory" pointing it is not applied. A continuous pointing angle will introduce a factor based on the pointing angle that is bounded by these two cases, but allows for continuous application of the final private method (not just a toggle).

## Framework for Standard Objects

Currently the fundamental objects (observatory, camera, pupil, satellites) must be initialized by arbitrary inputs provided by the user. The directive is to create a framework to define these objects as a an input file (YAML, JSON, config, non-exhaustive list of possibilities), that can be used to initialize objects from a factory method based on an object name/label.

Important context:
- `Pupil` class has some old machinery for initializing from a configuration, that was written to also be able to specify the geometric type of the pupil. This is a reference for intent, not necessarily the actual code implementation (it depends on a config file and may be a poor design decision).
