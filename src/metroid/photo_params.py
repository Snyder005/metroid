from dataclasses import dataclass

import astropy.units as u

from metroid.utils.quantities import Area, Gain, QuantumEfficiency, Time


@dataclass
class PhotometricParameters:
    exptime: Time
    gain: Gain
    area: Area
    qe: QuantumEfficiency = 1.0 * u.electron / u.ph
