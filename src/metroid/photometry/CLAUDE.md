# `metroid.photometry`

Radiometry layer: turning spectra and bandpasses into photon flux, energy
flux, and detector ADU. Wraps `speclite.filters` for the heavy lifting.

## Modules

- **`throughput.py` — `ThroughputCurve`.** Fractional transmission vs.
  wavelength, backed by a `speclite.filters.FilterResponse`. Constructors:
  `__init__(wavelength, throughput, meta)`, `from_filter_response(fr)`,
  `load_filter(name)` (loads a speclite-registered filter, e.g.
  `"lsst2023-g"`). Key methods:
  - `calculate_photon_flux` / `calculate_energy_flux` — convolve the curve
    with an SED (photon- or energy-weighted).
  - `calculate_adu(brightness_spec, photo_params)` — photon flux scaled to ADU.
  - `calculate_ab_magnitude(sed)`.
  - `brightness_spec` may be a `Sed` *or* a `float` AB magnitude;
    `_ensure_sed` converts a float by scaling a flat-AB reference SED by
    `10 ** (-0.4 * mag)`.
  - Immutability: the underlying `FilterResponse` wavelength/response arrays are
    frozen (`_freeze_filter_response` sets `flags.writeable = False`).
- **`sed.py` — `Sed`.** A spectral energy distribution (`wavelength`,
  `flambda`). `for_ab_magnitudes()` builds a flat reference SED
  (`flambda = _ab_constant / wavelength**2`) used as the magnitude reference.
  Wavelengths are validated/normalized to Angstrom via
  `speclite.filters.validate_wavelength_array`.
- **`photo_params.py` — `PhotometricParameters`.** A frozen, unit-validated
  dataclass (`@validated_dataclass(frozen=True)`) holding `exptime`, `gain`,
  `area`, `qe`. This is the one dataclass-style validated type in the codebase.
- **`conversions.py`** — free functions: `energy_flux_to_radiance(flux,
  solid_angle)` and `photon_flux_to_adu(photon_flux, photo_params)`.

## Dependencies

- External: `speclite` (`FilterResponse`, `load_filter`, `_ab_constant`,
  `validate_wavelength_array`), `astropy.units`.
- Internal: `metroid.utils.decorators.enforce_units` / `validated_dataclass`
  and the quantity aliases in `metroid.utils.quantities`.

## Known issues / gotchas

- `_ab_constant` and `validate_wavelength_array` are speclite *private/internal*
  API; upgrades to speclite could break `sed.py`.
- **`_freeze_filter_response` mutates speclite's global filter cache (#19).**
  `load_filter` returns a *shared, cached* `FilterResponse` whose ndarrays are
  shared across callers; setting `flags.writeable = False` on them leaks
  read-only state into every speclite consumer in the process.
  `from_filter_response(fr)` likewise freezes the caller's object in place. The
  fix is to copy the wavelength/response arrays before freezing so the freeze
  only touches arrays this `ThroughputCurve` owns.
- **`int` AB magnitudes are rejected (#20).** `calculate_energy_flux` /
  `calculate_adu` advertise `float | int | Sed`, but `_ensure_sed` guards with
  `isinstance(..., float)`, so an integer magnitude raises `TypeError`. The
  three public methods also disagree on their hints (`calculate_photon_flux`
  omits `int`). Note `bool` subclasses `int` — any numeric guard should decide
  deliberately whether `True`/`False` are valid.
- **`from_filter_response` hand-mangles `_ThroughputCurve__fr`** via
  `cls.__new__(cls)`, which trips mypy (`"Self" has no attribute ...`) and is
  fragile under rename. Routing all three constructors through one private
  initializer would fix this and is the natural home for the copy-before-freeze
  fix above.
- `calculate_ab_magnitude` and `_convolve` bypass `@enforce_units`, unlike the
  rest of the package's "every public method validates units" pattern.
- `qe` in `PhotometricParameters` is a scalar `QuantumEfficiency[Scalar]`, so
  wavelength-dependent detector QE is collapsed to a band average — a known
  first-approximation limitation, superseded once throughput composition lands
  (see roadmap).

## Optimization notes (within the current framework)

- **Float/AB path can skip the convolution.** Convolution is linear in
  `flambda` and the flat reference SED's photon flux *is* `ab_zeropoint`, so
  `calculate_photon_flux(mag) == ab_zeropoint * 10 ** (-0.4 * mag)`. The float
  path can return this directly instead of convolving an 8501-point SED.
- **The reference SED is rebuilt every float call** (`Sed.for_ab_magnitudes()`
  in `_ensure_sed`, ~1 ms/call of pure waste). Cache it once (module constant /
  `lru_cache` / cached class attribute).
- **`_convolve` passes both `sed.flambda` and `units=sed.flambda.unit`** —
  redundant; pass `.value` with `units=`, or the Quantity alone.

## Roadmap toward LEO-object photometry

The package is currently the **radiometry layer**: SED × bandpass →
photon/energy flux → ADU. Satellite/debris photometry adds geometry, solar
illumination, and trailing. Extend via the existing patterns (new `Spec` +
generic alias in `utils/quantities.py`; `@enforce_units` free functions in
`conversions.py`; `Constraint` classes), keeping the boundary clean: photometry
*consumes* scalar geometry (range, solid angle, angular rate) computed by the
orbital subpackage, and spatial smearing belongs in `profiles`.

- **Range scaling** — `radiant_intensity_to_flux(intensity, range)` closing the
  inverse-square chain (`RADIANT_INTENSITY` and `ENERGY_FLUX` already exist).
- **Solar-reflection SEDs** — `Sed.for_solar()` plus a reflectance/albedo hook;
  satellites reflect sunlight rather than emit it.
- **Phase angle / standard magnitude** — pluggable phase function and
  range/phase normalization so predictions match published satellite
  photometry.
- **Trailed photometry** — distribute total flux/ADU along the streak using
  angular rate and exptime to get ADU-per-pixel; trail *geometry* stays in
  `profiles`.
- **Atmospheric extinction + composable system throughput** — model QE, mirror
  reflectivity, and airmass-dependent extinction as `ThroughputCurve`s and
  compose (`__mul__` / `compose([...])`); this is the principled replacement
  for scalar `qe`.
- **Detector realism** (saturation / non-linearity) and **sky-background SNR**
  for detectability of faint trails.
