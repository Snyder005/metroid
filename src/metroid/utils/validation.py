from collections.abc import Mapping


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
        raise TypeError(f"name must be 'str'")

    try:
        value = config[name]
    except KeyError:
        raise ValueError(f"config is missing required field {name}")

    if not isinstance(value, dtype):
        raise TypeError(f"value must be '{dtype.__name__}'")

    return value
