from astropy import units as u
import numpy as np
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

    if not quantity.unit.is_equivalent(spec.default):
        raise ValueError(f"invalid unit for {spec.name}: {quantity.unit}")

    quantity = quantity.to(spec.default)
    if spec.typical_range is not None:
        vmin, vmax = spec.typical_range
        values = quantity.value
        if np.isscalar(values):
            if not (vmin <= values <= vmax):
                raise ValueError(f"{spec.name} value {values} is outside range {vmin}-{vmax}")

        else:
            if not np.all((values >= vmin) & (values <= vmax)):
                raise ValueError(f"{spec.name} contains values outside range {vmin}-{vmax}")

    return quantity
