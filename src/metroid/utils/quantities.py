"""Physical quantity specifications and runtime validation.

A :class:`QuantitySpec` reduces a physical quantity to its essential
identity - a name, a canonical unit, and allowed equivalencies - plus an
ordered list of pluggable :class:`Constraint` objects. Validation converts
to canonical units once and then runs every value-level constraint,
aggregating all failures into a single :class:`QuantityValidationError`.

New kinds of value-level check are added by writing a small constraint
class; the core :func:`check_quantity` validator never changes. The fluent
:class:`Spec` builder keeps the catalogue declarations terse.
"""

from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Protocol, Union, get_args, get_origin, runtime_checkable

import astropy.units as u
import numpy as np


class QuantityValidationError(ValueError):
    """All value-level validation failures for a single quantity, aggregated.

    Subclasses :class:`ValueError`, so existing ``except ValueError``
    handlers continue to catch it.

    Parameters
    ----------
    name : `str`
        The physical quantity name.
    problems : `list` [`str`]
        One human-readable message per failed constraint.
    """

    def __init__(self, name: str, problems: list[str]):
        self.name = name
        self.problems = problems
        super().__init__(f"{name} failed validation: {'; '.join(problems)}")


@runtime_checkable
class Constraint(Protocol):
    """A single, self-describing value-level check.

    A constraint inspects an already-unit-converted quantity and returns an
    error message if it is violated, or ``None`` if satisfied.
    """

    def check(self, quantity: u.Quantity, name: str) -> str | None:
        """Return an error message, or ``None`` if the constraint holds."""
        ...


@dataclass(frozen=True)
class Range:
    """Require every value to lie within an inclusive range.

    Parameters
    ----------
    vmin, vmax : `float`
        The inclusive bounds, in the spec's canonical unit.
    """

    vmin: float
    vmax: float

    def check(self, quantity: u.Quantity, name: str) -> str | None:
        value = quantity.value
        if not np.all((value >= self.vmin) & (value <= self.vmax)):
            return f"has values outside range {self.vmin}-{self.vmax}"
        return None


@dataclass(frozen=True)
class Finite:
    """Require every value to be finite (no NaN or inf)."""

    def check(self, quantity: u.Quantity, name: str) -> str | None:
        if not np.all(np.isfinite(quantity.value)):
            return "contains non-finite values (NaN or inf)"
        return None


@dataclass(frozen=True)
class QuantitySpec:
    """A physical quantity specification - unit identity plus constraints.

    Prefer the fluent :class:`Spec` builder for declarations.

    Parameters
    ----------
    name : `str`
        The physical quantity name.
    default : `astropy.units.Unit`
        The canonical unit.
    equivalencies : `list`, optional
        Unit equivalencies allowed during conversion.
    constraints : `tuple` [`Constraint`, ...], optional
        Ordered value-level constraints.
    """

    name: str
    default: u.Unit
    equivalencies: list = field(default_factory=list)
    constraints: tuple[Constraint, ...] = ()


class Spec:
    """Fluent builder producing an immutable :class:`QuantitySpec`.

    Example
    -------
    ``Spec("gain", u.electron / u.adu).ranged(0.1, 100).build()``
    """

    def __init__(self, name: str, default: u.Unit, equivalencies: list | None = None):
        self._spec = QuantitySpec(name, default, equivalencies or [])

    def _add(self, constraint: Constraint) -> "Spec":
        self._spec = replace(self._spec, constraints=self._spec.constraints + (constraint,))
        return self

    def ranged(self, vmin: float, vmax: float) -> "Spec":
        """Require values within ``[vmin, vmax]``."""
        return self._add(Range(vmin, vmax))

    def finite(self) -> "Spec":
        """Require finite values."""
        return self._add(Finite())

    def with_constraint(self, constraint: Constraint) -> "Spec":
        """Attach an arbitrary custom constraint."""
        return self._add(constraint)

    def build(self) -> QuantitySpec:
        """Return the assembled, immutable specification."""
        return self._spec


def check_quantity(quantity: u.Quantity, spec: QuantitySpec) -> u.Quantity:
    """Check that quantity has valid units and value.

    The unit is converted to the spec's canonical unit, then every
    value-level constraint is run and all failures are aggregated into a
    single :class:`QuantityValidationError`.

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

    Raises
    ------
    TypeError
        Raised if ``quantity`` or ``spec`` are invalid types.
    QuantityValidationError
        Raised if ``quantity`` fails one or more value-level constraints.
    ValueError
        Raised if the unit is not convertible to the canonical unit.
    """
    if not isinstance(spec, QuantitySpec):
        raise TypeError(f"{spec} must be 'metroid.utils.quantities.QuantitySpec'")

    if not isinstance(quantity, u.Quantity):
        raise TypeError(f"{spec.name} must be 'astropy.units.Quantity'")

    if not quantity.unit.is_equivalent(spec.default, equivalencies=spec.equivalencies):
        raise ValueError(f"invalid unit for {spec.name}: {quantity.unit}")

    quantity = quantity.to(spec.default, equivalencies=spec.equivalencies)

    problems = [msg for c in spec.constraints if (msg := c.check(quantity, spec.name)) is not None]
    if problems:
        raise QuantityValidationError(spec.name, problems)

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

        return _extract_spec(base)

    if origin is Union:
        for arg in get_args(annotation):
            spec = _extract_spec(arg)
            if spec:
                return spec

    return None


# ----------------------------------------------------------------------------
# Spec catalogue.  Declared with the fluent builder and reads as a table of
# "what each quantity is".  Range limits are intentionally omitted here and
# will be reworked per physical quantity on a case-by-case basis.
# ----------------------------------------------------------------------------

WAVELENGTH = Spec("wavelength", u.AA).build()
"""The wavelength specification."""

GEOMETRY_LENGTH = Spec("geometry_length", u.m).build()
"""The geometry length specification."""

ORBITAL_DISTANCE = Spec("orbital_distance", u.km).build()
"""The orbital distance specification."""

AREA = Spec("area", u.m**2).build()
"""The area specification."""

TIME = Spec("time", u.s).build()
"""The time specification."""

VELOCITY = Spec("velocity", u.m / u.s).build()
"""The velocity specification."""

ANGLE = Spec("angle", u.deg).build()
"""The angle specification."""

SOLID_ANGLE = Spec("solid_angle", u.sr, u.dimensionless_angles()).build()
"""The solid angle specification."""

ANGULAR_VELOCITY = Spec("angular_velocity", u.rad / u.s, u.dimensionless_angles()).build()
"""The angular velocity specification."""

ADU = Spec("adu", u.adu).build()
"""The adu specification."""

GAIN = Spec("gain", u.electron / u.adu).build()
"""The gain specification."""

QUANTUM_EFFICIENCY = Spec("qe", u.electron / u.ph).build()
"""The quantum efficiency specification."""

PIXEL_SCALE = Spec("pixel_scale", u.arcsec / u.pix).build()
"""The pixel scale specification."""

FRACTION = Spec("fraction", u.dimensionless_unscaled).build()
"""The throughput specification."""

SPECTRAL_FLUX_DENSITY = Spec("spectral_flux_density", u.erg / (u.s * u.cm**2 * u.AA)).build()
"""The wavelength spectral flux density specification."""

PHOTON_FLUX = Spec("photon_flux", u.ph / (u.s * u.m**2), [(u.ph, None)]).build()
"""The spectral photon flux density specification."""

ENERGY_FLUX = Spec("energy_flux", u.erg / (u.s * u.m**2)).build()
"""The energy flux density (irradiance) specification."""

RADIANCE = Spec("radiance", u.W / (u.sr * u.m**2)).build()
"""The radiance specification."""

RADIANT_INTENSITY = Spec("radiant_intensity", u.W / u.sr).build()
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
