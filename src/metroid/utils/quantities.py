from typing import Annotated, Any, Union, get_args, get_origin
from astropy import units as u


class QuantitySpec:
    """A quantity specification."""

    def __init__(self, name: str, default: u.Unit, typical_range: tuple[float, float] | None = None):
        self.name = name
        self.default = default
        self.typical_range = typical_range


def extract_spec(annotation: Any) -> QuantitySpec | None:
    """Extract the quantity specification from a type hint.

    Parameters
    ----------
    annotation : typehint
        The type hint.

    Returns
    -------
    spec : `metroid.utils.quantities.QuantitySpec` or None
        The extracted quantity specification.
    """
    if annotation is None:
        return None

    origin = get_origin(annotation)
    if origin is Annotated:
        base, *meta = get_args(annotation)
        for m in meta:
            if isinstance(m, QuantitySpec):
                return m

    if origin is Union:
        for arg in get_args(annotation):
            spec = extract_spec(arg)
            if spec:
                return spec

    return None


WAVELENGTH = QuantitySpec("wavelength", u.AA)
"""The wavelength specification."""

GEOMETRYLENGTH = QuantitySpec("geometry_length", u.m, typical_range=(1e-3, 1e3))
"""The geometry length specification."""

ORBITALDISTANCE = QuantitySpec("orbital_distance", u.km, typical_range=(1e2, 1e6))
"""The orbital distance specification."""

AREA = QuantitySpec("area", u.m**2, typical_range=(1e-6, 1e6))
"""The area specification."""

TIME = QuantitySpec("time", u.s)
"""The time specification."""

VELOCITY = QuantitySpec("velocity", u.m / u.s)
"""The velocity specification."""

ANGLE = QuantitySpec("angle", u.deg)
"""The angle specification."""

SOLIDANGLE = QuantitySpec("solid_angle", u.sr)
"""The solid angle specification."""

ANGULARVELOCITY = QuantitySpec("angular_velocity", u.rad / u.s)
"""The angular velocity specification."""

ADU = QuantitySpec("adu", u.adu)
"""The adu specification."""

GAIN = QuantitySpec("gain", u.electron / u.adu, typical_range=(1e-1, 1e2))
"""The gain specification."""

PIXELSCALE = QuantitySpec("pixel_scale", u.arcsec / u.pix, typical_range=(1e-2, 1e1))
"""The pixel scale specification."""

THROUGHPUT = QuantitySpect("throughput", u.dimensionless_unscaled)

FLUXDENSITY = QuantitySpec("flux_density", u.erg / (u.s * u.cm**2 * u.AA))
"""The flux density specification."""

PHOTONFLUXDENSITY = QuantitySpec("photon_flux_density", u.electron / (u.m**2 * u.s))

RADIANTINTENSITY = QuantitySpec("radiant_intensity", u.W / u.sr)
"""The radiant intensity specification."""

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
Throughput = Annotated[u.Quantity, THROUGHPUT]
FluxDensity = Annotated[u.Quantity, FLUXDENSITY]
RadiantIntensity = Annotated[u.Quantity, RADIANTINTENSITY]
