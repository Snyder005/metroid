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

import types
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Protocol, TypeAliasType, Union, get_args, get_origin, runtime_checkable

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


# ----------------------------------------------------------------------------
# Shape: a separate axis from the physical spec.  Scalar-vs-array is a computer
# representation, not a physical property, so it lives outside QuantitySpec and
# is supplied at the annotation site via a generic parameter (e.g. Time[Scalar])
# and to check_quantity as the ``shape`` argument.
# ----------------------------------------------------------------------------


@runtime_checkable
class ShapeKind(Protocol):
    """A check on the array-shape of a quantity (scalar vs. array).

    Mirrors :class:`Constraint` but is kept distinct so that shape is not
    confused with a physical, value-level constraint.
    """

    def check(self, quantity: u.Quantity, name: str) -> str | None:
        """Return an error message, or ``None`` if the shape is allowed."""
        ...


@dataclass(frozen=True)
class _Scalar:
    """Require the quantity to be a single scalar value."""

    def check(self, quantity: u.Quantity, name: str) -> str | None:
        if not quantity.isscalar:
            return f"must be scalar, got shape {quantity.shape}"
        return None


@dataclass(frozen=True)
class _Array:
    """Require the quantity to be array-valued (non-scalar)."""

    def check(self, quantity: u.Quantity, name: str) -> str | None:
        if quantity.isscalar:
            return "must be an array, got a scalar"
        return None


@dataclass(frozen=True)
class _AnyShape:
    """Impose no shape restriction (scalar or array both allowed)."""

    def check(self, quantity: u.Quantity, name: str) -> str | None:
        return None


# Stateless singletons - one instance of each shape suffices.
SCALAR: ShapeKind = _Scalar()
ARRAY: ShapeKind = _Array()
ANY_SHAPE: ShapeKind = _AnyShape()


class Scalar:
    """Marker type for a scalar quantity. Use as a subscript: ``Time[Scalar]``."""


class Array:
    """Marker type for an array quantity. Use as a subscript: ``Time[Array]``."""


class AnyShape:
    """Marker type imposing no shape restriction (the default for a bare alias)."""


_SHAPE_BY_MARKER: dict[Any, ShapeKind] = {
    Scalar: SCALAR,
    Array: ARRAY,
    AnyShape: ANY_SHAPE,
}


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


def check_quantity(quantity: u.Quantity, spec: QuantitySpec, shape: ShapeKind = ANY_SHAPE) -> u.Quantity:
    """Check that quantity has valid units, value, and shape.

    The unit is converted to the spec's canonical unit, then every
    value-level constraint and the shape restriction are run and all
    failures are aggregated into a single :class:`QuantityValidationError`.

    Parameters
    ----------
    quantity : `astropy.units.Quantity`
        The quantity to check.
    spec : `metroid.utils.quantities.QuantitySpec`
        The quantity specification.
    shape : `metroid.utils.quantities.ShapeKind`, optional
        The shape restriction (``SCALAR``, ``ARRAY``, or ``ANY_SHAPE``).
        Defaults to ``ANY_SHAPE`` (scalar or array both allowed).

    Returns
    -------
    quantity : `astropy.units.Quantity`
        The quantity in the specified default units.

    Raises
    ------
    TypeError
        Raised if ``quantity`` or ``spec`` are invalid types.
    QuantityValidationError
        Raised if ``quantity`` fails one or more value-level or shape checks.
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
    if (msg := shape.check(quantity, spec.name)) is not None:
        problems.append(msg)
    if problems:
        raise QuantityValidationError(spec.name, problems)

    return quantity


def _spec_from_annotated(annotation: Any) -> QuantitySpec | None:
    """Pull a :class:`QuantitySpec` out of an ``Annotated[...]`` value."""
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, QuantitySpec):
                return meta
    return None


def _extract_spec(annotation: Any) -> tuple[QuantitySpec | None, ShapeKind]:
    """Extract the ``(spec, shape)`` pair from a type hint.

    Recurses through generic quantity aliases (``Time``, ``Time[Scalar]``),
    bare ``Annotated`` hints, and unions (e.g. ``Time[Scalar] | None``). The
    shape defaults to ``ANY_SHAPE`` when no shape marker is supplied.

    Parameters
    ----------
    annotation : typehint
        The type hint.

    Returns
    -------
    spec : `metroid.utils.quantities.QuantitySpec` or None
        The extracted quantity specification, or ``None`` if absent.
    shape : `metroid.utils.quantities.ShapeKind`
        The shape restriction; ``ANY_SHAPE`` if unspecified.
    """
    if annotation is None:
        return None, ANY_SHAPE

    # Bare generic alias, e.g. ``Time`` - a TypeAliasType with unbound param.
    if isinstance(annotation, TypeAliasType):
        return _spec_from_annotated(annotation.__value__), ANY_SHAPE

    origin = get_origin(annotation)

    # Subscripted alias, e.g. ``Time[Scalar]`` - origin is the TypeAliasType,
    # args carry the supplied shape marker.
    if isinstance(origin, TypeAliasType):
        spec = _spec_from_annotated(origin.__value__)
        args = get_args(annotation)
        shape = _SHAPE_BY_MARKER.get(args[0], ANY_SHAPE) if args else ANY_SHAPE
        return spec, shape

    # A directly written ``Annotated`` (not via an alias).
    if origin is Annotated:
        return _spec_from_annotated(annotation), ANY_SHAPE

    # Optional / Union, e.g. ``Time[Scalar] | None``; first hit wins.
    # ``X | Y`` (PEP 604) has origin ``types.UnionType``; ``Union[X, Y]`` has
    # origin ``typing.Union``.
    if origin is Union or origin is types.UnionType:
        for arg in get_args(annotation):
            spec, shape = _extract_spec(arg)
            if spec is not None:
                return spec, shape

    return None, ANY_SHAPE


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
"""The photon flux density specification."""

ENERGY_FLUX = Spec("energy_flux", u.erg / (u.s * u.m**2)).build()
"""The energy flux density (irradiance) specification."""

RADIANCE = Spec("radiance", u.W / (u.sr * u.m**2)).build()
"""The radiance specification."""

RADIANT_INTENSITY = Spec("radiant_intensity", u.W / u.sr).build()
"""The radiant intensity specification."""

# Generic type aliases - one per physical quantity, parameterized by shape.
# The bare alias (``Time``) means "any shape"; ``Time[Scalar]`` / ``Time[Array]``
# add a shape restriction.  The wrapped type stays ``u.Quantity``, so static
# type checkers and editors still treat annotated values as quantities.

type Wavelength[Sh] = Annotated[u.Quantity, WAVELENGTH, Sh]
type GeometryLength[Sh] = Annotated[u.Quantity, GEOMETRY_LENGTH, Sh]
type OrbitalDistance[Sh] = Annotated[u.Quantity, ORBITAL_DISTANCE, Sh]
type Area[Sh] = Annotated[u.Quantity, AREA, Sh]
type Time[Sh] = Annotated[u.Quantity, TIME, Sh]
type Velocity[Sh] = Annotated[u.Quantity, VELOCITY, Sh]
type Angle[Sh] = Annotated[u.Quantity, ANGLE, Sh]
type SolidAngle[Sh] = Annotated[u.Quantity, SOLID_ANGLE, Sh]
type AngularVelocity[Sh] = Annotated[u.Quantity, ANGULAR_VELOCITY, Sh]
type Adu[Sh] = Annotated[u.Quantity, ADU, Sh]
type Gain[Sh] = Annotated[u.Quantity, GAIN, Sh]
type QuantumEfficiency[Sh] = Annotated[u.Quantity, QUANTUM_EFFICIENCY, Sh]
type PixelScale[Sh] = Annotated[u.Quantity, PIXEL_SCALE, Sh]
type Fraction[Sh] = Annotated[u.Quantity, FRACTION, Sh]
type SpectralFluxDensity[Sh] = Annotated[u.Quantity, SPECTRAL_FLUX_DENSITY, Sh]
type PhotonFlux[Sh] = Annotated[u.Quantity, PHOTON_FLUX, Sh]
type EnergyFlux[Sh] = Annotated[u.Quantity, ENERGY_FLUX, Sh]
type Radiance[Sh] = Annotated[u.Quantity, RADIANCE, Sh]
type RadiantIntensity[Sh] = Annotated[u.Quantity, RADIANT_INTENSITY, Sh]
