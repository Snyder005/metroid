from typing import Annotated
from astropy import units as u

class QuantitySpec:
    def __init__(self, name: str, default: u.Unit, typical_range: tuple[float, float] | None = None):
        self.name = name
        self.default = default
        self.typical_range = typical_range

def extract_spec(annotation) -> QuantitySpec | None:
    if annotation is None:
        return None

    if get_origin(annotation) is Annotated:
        base, *meta = get_args(annotation)
        for m in meta: 
            if isinstance(m, QuantitySpec):
                return m.kind

    if origin is Union:
        for arg in get_args(annotation):
            spec = extract_spec(arg)
            if spec:
                return spec
            
    return None

WAVELENGTH = QuantitySpec("wavelength", u.AA)
GEOMETRYLENGTH = QuantitySpec("geometry_length", u.m, typical_range=(1e-3, 1e3))
ORBITALDISTANCE = QuantitySpec("orbital_distance", u.km, typical_range=(1e2, 1e6))
AREA = QuantitySpec("area", u.m ** 2, typical_range=(1e-6, 1e6))
TIME = QuantitySpec("time", u.s)
VELOCITY = QuantitySpec("velocity", u.m / u.s)
ANGLE = QuantitySpec("angle", u.deg)
SOLIDANGLE = QuantitySpec("solid_angle", u.sr)
ANGULARVELOCITY = QuantitySpec("angular_velocity", u.rad / u.s)
ADU = QuantitySpec("adu", u.adu)
GAIN = QuantitySpec("gain", u.electron / u.adu, typical_range=(1e-1, 1e2))
PIXELSCALE = QuantitySpec("pixel_scale", u.arcsec / u.pix, typical_range=(1e-2, 1e1))
RADIANTINTENSITY = QuantitySpec("radiant_intensity", u.W / u.sr)

Wavelength = Annotated[u.Quantity, WAVELENGTH]
GeometryLength = Annotated[u.Quantity, GEOMETRYLENGTH]
OrbitalDistance = Annotated[u.Quantity, ORBITALDISTANCE]
Area = Annotated[u.Quantity, AREA]
Time = Annotated[u.Quantity, TIME]
Velocity = Annotated[u.Quantity, VELOCITY]
Angle = Annotated[u.Quantity, ANGLE]
SolidAngle = Annotated[u.Quantity, SOLIDANGLE]
AngularVelocity = Annotated[u.Quantity, ANGULARVELOCITY]
Adu = Annotated[u.Quantity, ADU]
Gain = Annotated[u.Quantity, GAIN]
PixelScale = Annotated[u.Quantity, PIXELSCALE]
RadiantIntensity = Annotated[u.Quantity, RADIANTINTENSITY]
