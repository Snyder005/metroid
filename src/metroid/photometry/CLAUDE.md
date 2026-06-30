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
