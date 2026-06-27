import astropy.units as u

from metroid.utils.decorators import validated_dataclass
from metroid.utils.quantities import Area, Gain, QuantumEfficiency, Time


@validated_dataclass(frozen=True)
class PhotometricParameters:
    """Photometric parameters for an observation."""

    exptime: Time
    gain: Gain
    area: Area
    qe: QuantumEfficiency = 1.0 * u.electron / u.ph
