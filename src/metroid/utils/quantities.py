from typing import Annotated, Any, Union, get_args, get_origin

import astropy.units as u
import numpy as np


class QuantitySpec:
    """A quantity specification."""

    def __init__(
        self,
        name: str,
        default: u.Unit,
        typical_range: tuple[float, float] | None = None,
        equivalencies: list | None = None,
    ):
        self.name = name
        self.default = default
        self.typical_range = typical_range
        self.equivalencies = equivalencies or []


def check_quantity(quantity: u.Quantity, spec: QuantitySpec) -> u.Quantity:
    """Check that quantity has valid units and value.

    Parameters
    ----------
    quantity : `astropy.units.Quantity`
        The quantity to check.
    spec : `metroid.utils.quantities.QuantitySpec`
        The quantity specification.

    Returns
    -------
    quantity : `astropy.units.Quantity`
        The quantity in the specified default units.
    """
    if not isinstance(spec, QuantitySpec):
        raise TypeError(f"{spec} must be 'metroid.utils.quantities.QuantitySpec'")

    if not isinstance(quantity, u.Quantity):
        raise TypeError(f"{spec.name} must be 'astropy.units.Quantity'")

    if not quantity.unit.is_equivalent(spec.default, equivalencies=spec.equivalencies):
        raise ValueError(f"invalid unit for {spec.name}: {quantity.unit}")

    quantity = quantity.to(spec.default, equivalencies=spec.equivalencies)
    if spec.typical_range is not None:
        value = quantity.value
        vmin, vmax = spec.typical_range
        if np.isscalar(value):
            if not (vmin <= value <= vmax):
                raise ValueError(f"{spec.name} value {value} is outside range {vmin}-{vmax}")

        else:
            if not np.all((value >= vmin) & (value <= vmax)):
                raise ValueError(f"{spec.name} contains values outside range {vmin}-{vmax}")

    return quantity


def _extract_spec(annotation: Any) -> QuantitySpec | None:
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
            spec = _extract_spec(arg)
            if spec:
                return spec

    return None


WAVELENGTH = QuantitySpec("wavelength", u.AA)
"""The wavelength specification."""

GEOMETRY_LENGTH = QuantitySpec("geometry_length", u.m, typical_range=(1e-3, 1e3))
"""The geometry length specification."""

ORBITAL_DISTANCE = QuantitySpec("orbital_distance", u.km, typical_range=(1e2, 1e6))
"""The orbital distance specification."""

AREA = QuantitySpec("area", u.m**2, typical_range=(1e-6, 1e6))
"""The area specification."""

TIME = QuantitySpec("time", u.s, typical_range=(0.0, 1e5))
"""The time specification."""

VELOCITY = QuantitySpec("velocity", u.m / u.s)
"""The velocity specification."""

ANGLE = QuantitySpec("angle", u.deg)
"""The angle specification."""

SOLID_ANGLE = QuantitySpec("solid_angle", u.sr, equivalencies=u.dimensionless_angles())
"""The solid angle specification."""

ANGULAR_VELOCITY = QuantitySpec("angular_velocity", u.rad / u.s, equivalencies=u.dimensionless_angles())
"""The angular velocity specification."""

ADU = QuantitySpec("adu", u.adu)
"""The adu specification."""

GAIN = QuantitySpec("gain", u.electron / u.adu, typical_range=(1e-1, 1e2))
"""The gain specification."""

QUANTUM_EFFICIENCY = QuantitySpec("qe", u.electron / u.ph, typical_range=(1e0, 1e2))
"""The quantum efficiency specification."""

PIXEL_SCALE = QuantitySpec("pixel_scale", u.arcsec / u.pix, typical_range=(1e-2, 1e1))
"""The pixel scale specification."""

FRACTION = QuantitySpec("fraction", u.dimensionless_unscaled, typical_range=(0.0, 1.0))
"""The throughput specification."""

SPECTRAL_FLUX_DENSITY = QuantitySpec("spectral_flux_density", u.erg / (u.s * u.cm**2 * u.AA))
"""The wavelength spectral flux density specification."""

PHOTON_FLUX = QuantitySpec("photon_flux", u.ph / (u.s * u.m**2), equivalencies=[(u.ph, None)])
"""The spectral photon flux density specification."""

ENERGY_FLUX = QuantitySpec("energy_flux", u.erg / (u.s * u.m**2))
"""The energy flux density (irradiance) specification."""

RADIANCE = QuantitySpec("radiance", u.W / (u.sr * u.m**2))
"""The radiance specification."""

RADIANT_INTENSITY = QuantitySpec("radiant_intensity", u.W / u.sr)
"""The radiant intensity specification."""

Wavelength = Annotated[u.Quantity, WAVELENGTH]
GeometryLength = Annotated[u.Quantity, GEOMETRY_LENGTH]
OrbitalDistance = Annotated[u.Quantity, ORBITAL_DISTANCE]
Area = Annotated[u.Quantity, AREA]
Time = Annotated[u.Quantity, TIME]
Velocity = Annotated[u.Quantity, VELOCITY]
Angle = Annotated[u.Quantity, ANGLE]
SolidAngle = Annotated[u.Quantity, SOLID_ANGLE]
AngularVelocity = Annotated[u.Quantity, ANGULAR_VELOCITY]
Adu = Annotated[u.Quantity, ADU]
Gain = Annotated[u.Quantity, GAIN]
QuantumEfficiency = Annotated[u.Quantity, QUANTUM_EFFICIENCY]
PixelScale = Annotated[u.Quantity, PIXEL_SCALE]
Fraction = Annotated[u.Quantity, FRACTION]
SpectralFluxDensity = Annotated[u.Quantity, SPECTRAL_FLUX_DENSITY]
PhotonFlux = Annotated[u.Quantity, PHOTON_FLUX]
EnergyFlux = Annotated[u.Quantity, ENERGY_FLUX]
Radiance = Annotated[u.Quantity, RADIANCE]
RadiantIntensity = Annotated[u.Quantity, RADIANT_INTENSITY]
