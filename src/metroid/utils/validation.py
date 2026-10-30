import astropy.units as u
from collections.abc import Mapping

from ..typing.aliases import ANY_SHAPE, QuantitySpec, QuantityValidationError, ShapeKind


def check_quantity(quantity: u.Quantity, spec: QuantitySpec, shape: ShapeKind = ANY_SHAPE) -> u.Quantity:
    """Check that quantity has valid units, value, and shape.

    The unit is converted to the spec's canonical unit, then every
    value-level constraint and the shape restriction are run and all
    failures are aggregated into a single `QuantityValidationError`.

    Parameters
    ----------
    quantity : `astropy.units.Quantity`
        The quantity to check.
    spec : `metroid.utils.quantities.QuantitySpec`
        The quantity specification.Q
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


def get_field_value[T](config: Mapping[str, object], name: str, dtype: type[T]) -> T:
    """Get value from a configuration field.

    Parameters
    ----------
    config : `dict`
        A configuration dictionary of fields each consisting of a name (`str`)
        and value (any type).
    name : `str`
        The configuration field name.
    dtype : type(object)
        The Python data type of the value.

    Returns
    -------
    value : object
        The value corresponding to the field name.

    Raises
    ------
    ValueError
        Raised if the required field does not exist.
    TypeError
        Raised if ``name`` or ``value`` is an invalid type.
    """
    if not isinstance(name, str):
        raise TypeError("name must be 'str'")

    try:
        value = config[name]
    except KeyError:
        raise ValueError(f"config is missing required field '{name}'")

    if not isinstance(value, dtype):
        raise TypeError(f"value must be '{dtype.__name__}'")

    return value
