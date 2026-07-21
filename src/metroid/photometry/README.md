# photometry

## Overview

The radiometry layer turns spectra and bandpasses into photon flux,
energy flux, and detector ADU. It wraps `speclite.filters` for filter
response handling and exposes a small set of classes and free functions
as the public interface.

## Architecture

`ThroughputCurve` represents fractional transmission as a function of
wavelength, backed by a private `speclite.filters.FilterResponse`.
Three constructors are provided: `__init__` (wavelength and throughput
arrays), `from_filter_response` (from an existing `FilterResponse`),
and `load_filter` (by filter name). Methods on the class
(`calculate_photon_flux`, `calculate_energy_flux`, `calculate_adu`,
`calculate_ab_magnitude`) accept a `brightness_spec` that is either a
`Sed` instance or a numeric AB magnitude. When a numeric magnitude is
supplied, `_ensure_sed` constructs a flat-AB reference `Sed` and scales
its flux by `10**(-0.4 * mag)` before convolving.

`Sed` holds a wavelength array and a spectral flux density array
(`flambda`). The `for_ab_magnitudes()` classmethod builds the flat-AB
reference SED used internally by `ThroughputCurve._ensure_sed`.

`PhotometricParameters` is a frozen, unit-validated dataclass (using
`validated_dataclass` from `utils/decorators.py`) holding `exptime`,
`gain`, `area`, and `qe`. It is the one dataclass-style validated type
in the package.

`conversions.py` provides free functions `energy_flux_to_radiance` and
`photon_flux_to_adu`, which are also the computation kernels called by
`ThroughputCurve.calculate_adu`.

## Design Decisions

**Filter-response ownership.** `speclite.filters.load_filter` returns
a shared, cached `FilterResponse` whose underlying arrays are reused
across callers. `_adopt_filter_response` calls `copy.deepcopy` on the
`FilterResponse` to obtain independent arrays and to bypass
`FilterResponse.__init__`, preventing re-registration into the speclite
global cache. It then freezes the private copy by setting
`flags.writeable = False`. All three `ThroughputCurve` constructors
route through this method, so a `ThroughputCurve` only ever freezes
arrays it exclusively owns.

**Scalar QE.** The `qe` field in `PhotometricParameters` is a scalar
quantity, meaning wavelength-dependent detector quantum efficiency is
collapsed to a band average. This is a deliberate first-approximation
limitation.

## Invariants

`sed.py` depends on speclite private/internal API (`_ab_constant`,
`validate_wavelength_array`). A speclite upgrade could silently break
this module (tracked in issue #26).
