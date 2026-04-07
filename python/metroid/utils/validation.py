from astropy import units as u
import operator

def check_quantity(
        quantity: u.Quantity, 
        unit: u.Unit, 
        vmin: float = None, 
        vmax: float = None, 
        inclusive_min: bool = False, 
        inclusive_max: bool = False,
    ) -> u.Quantity:
    """Performs a sequence of checks on an astropy Quantity.

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
    is not isinstance(quantity, u.Quantity):
        raise TypeError(f"must be 'astropy.units.Quantity'")

    if not quantity.unit.is_equivalent(unit):
        raise ValueError(f"{quantity.unit} not equivalent with {unit}")

    if vmin is not None:
        min_op, op = (operator.lt, '>=') if inclusive_min else (operator.le, '>')
        if min_op(quantity, vmin*unit):
            raise ValueError(f"{quantity} must be {op} {vmin*unit}")

    if vmax is not None: 
        max_op, op = (operator.gt, '<=') if inclusive_min else (operator.ge, '<')
        if max_op(quantity, vmax*unit):
            raise ValueError(f"{quantity} must be {op} {vmax*unit}")

    return quantity
