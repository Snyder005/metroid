import astropy.units as u

import metroid.utils.quantities as q
from metroid.utils.decorators import validated_dataclass
from metroid.utils.quantities import Area, Gain, QuantumEfficiency, Scalar, Time


@validated_dataclass(frozen=True)
class PhotometricParameters:
    """Photometric parameters for an observation."""

    exptime: Time[Scalar]
    gain: Gain[Scalar]
    area: Area[Scalar]
    qe: QuantumEfficiency[Scalar] = 1.0 * u.electron / u.ph
