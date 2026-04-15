from astropy import units as u
import operator
from typing import TypeVar, Any, Literal

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


def check_quantity(
    quantity: u.Quantity,
    unit: u.Unit,
    vmin: float = None,
    vmax: float = None,
    inclusive: Literal["both", "none", "min", "max"] = "none",
) -> u.Quantity:
    """Perform a sequence of checks on an astropy Quantity.

    Parameters
    ----------
    quantity : `astropy.units.Quantity`
        A quantity to check.
    unit : `astropy.units.Unit`
        The expected unit of the quantity.
    vmin : `float`, optional
        The minimum limit in the expected unit (None, by default).
    vmax : `float`, optional
        The maximum limit in the expected unit (None, by default).
    inclusive : {'both', 'none', 'min', 'max'}, optional
        Specify which bounds are inclusive ('none', by default).

    Returns
    -------
    quantity : `astropy.units.Quantity`
        The verified quantity.

    Raises
    ------
    TypeError
        Raised if `quantity` is an invalid type.
    ValueError
        Raised if `quantity` has an invalid unit or value.
    """
    if not isinstance(quantity, u.Quantity):
        raise TypeError(f"must be 'astropy.units.Quantity'")

    if not quantity.unit.is_equivalent(unit):
        raise ValueError(f"{quantity.unit} not equivalent with {unit}")

    include_min = inclusive in ("both", "min")
    include_max = inclusive in ("both", "max")

    if vmin is not None:
        min_op, op = (operator.lt, ">=") if include_min else (operator.le, ">")
        if min_op(quantity, vmin * unit):
            raise ValueError(f"{quantity} must be {op} {vmin * unit}")

    if vmax is not None:
        max_op, op = (operator.gt, "<=") if include_max else (operator.ge, "<")
        if max_op(quantity, vmax * unit):
            raise ValueError(f"{quantity} must be {op} {vmax * unit}")

    return quantity
