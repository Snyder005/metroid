from astropy import units as u
import operator
from typing import TypeVar, Any, Literal

from metroid.utils.quantities import QuantitySpec

T = TypeVar("T", str, float, list)


def get_field_value(config: dict[str, Any], name: str, dtype: type[T]) -> T:
    """Get value from a configuration field.

    Parameters
    ----------
    config : `dict`
        A configuration dictionary of fields each consisting of a name (`str`)
        and value (any type).
    name : `str`
        The configuration field name.
    dtype : type
        The Python data type of the value.

    Returns
    -------
    value : `str`, `float`, or `list`
        The value corresponding to the field name.

    Raises
    ------
    ValueError
        Raised if the required field does not exist.
    TypeError
        Raised if the name or the value is an invalid type.
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be 'str'")

    try:
        value = config[name]
    except KeyError:
        raise ValueError(f"missing required field '{name}'")

    if not isinstance(value, dtype):
        raise TypeError(f"value must be '{dtype.__name__}'")

    return value


def check_quantity(quantity: u.Quantity, spec: QuantitySpec) -> u.Quantity:

    if not isinstance(quantity, u.Quantity):
        raise TypeError(f"{spec.name} must be 'astropy.units.Quantity'")

    if not quantity.unit.is_equivalent(spec.default):
        raise ValueError(f"invalid unit for {spec.name}: {quantity.unit}")

    quantity = quantity.to(spec.default)

    if spec.range:
        vmin, vmax = spec.range
        if not (vmin <= quantity.value <= vmax):
            raise ValueError(f"{kind} value {quantity.value} is outside expected range {vmin}-{vmax}")

    return quantity
