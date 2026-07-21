# photometry/

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `throughput.py` | `ThroughputCurve` class wrapping a `speclite` `FilterResponse`; methods to calculate photon flux, energy flux, ADU, and AB magnitude from an SED or AB magnitude float | Implementing bandpass convolution; debugging photon/energy flux or ADU calculations; adding filter loading |
| `sed.py` | `Sed` class holding wavelength and spectral flux density arrays; `for_ab_magnitudes` factory for a flat AB reference SED | Implementing or modifying spectral energy distributions; debugging AB magnitude conversions |
| `photo_params.py` | `PhotometricParameters` frozen, unit-validated dataclass holding exposure time, gain, area, and quantum efficiency | Implementing observations requiring photometric scaling; debugging parameter construction |
| `conversions.py` | Free functions `energy_flux_to_radiance` and `photon_flux_to_adu` | Implementing flux unit conversions; adding new photometric conversion functions |
| `__init__.py` | Public exports: `ThroughputCurve`, `Sed`, `PhotometricParameters`, `energy_flux_to_radiance`, `photon_flux_to_adu` | Checking what `metroid.photometry` exposes |
