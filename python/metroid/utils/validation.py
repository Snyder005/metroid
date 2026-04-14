from astropy import units as u
import operator
from typing import TypeVar

T = TypeVar("T", str, float, list)


def get_field_value(config: dict[str, str | float | list], name: str, dtype: type[T]) -> T:
    """Get value from a configuration field.

    Parameters
    ----------
    config : `dict`
        A dictionary of configuration fields consisting of a name (`str`) and
        value (`str`, `float`, or `list`) pair.
    name : `str`
        The field name.
    dtype : type
        A Python data type (`str`, `float`, or `list`).

    Returns
    -------
    value : `str`, `float`, or `list`
        The value corresponding to the field name.

    Raises
    ------
    ValueError
        Raised if the required field does not exist.
    TypeError
        Raised if the value is an invalid type.
    """
    try:
        value = config[name]
    except KeyError:
        raise ValueError(f"missing required field '{name}'")

    if not isinstance(value, dtype):
        raise TypeError("value must be '{dtype.__name__}'")

    return value


def check_quantity(
    quantity: u.Quantity,
    unit: u.Unit,
    vmin: float = None,
    vmax: float = None,
    inclusive_min: bool = False,
    inclusive_max: bool = False,
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
    inclusive_min : `bool`, optional
        Include the minimum in the bounds if `True` (`False`, by default).
    inclusive_max : `bool`, optional
        Include the maximum in the bounds if `True` (`False`, by default).

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

    if vmin is not None:
        min_op, op = (operator.lt, ">=") if inclusive_min else (operator.le, ">")
        if min_op(quantity, vmin * unit):
            raise ValueError(f"{quantity} must be {op} {vmin * unit}")

    if vmax is not None:
        max_op, op = (operator.gt, "<=") if inclusive_min else (operator.ge, "<")
        if max_op(quantity, vmax * unit):
            raise ValueError(f"{quantity} must be {op} {vmax * unit}")

    return quantity
