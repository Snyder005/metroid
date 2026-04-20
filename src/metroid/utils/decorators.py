import inspect
from typing import Callable, ParamSpec, TypeVar, get_type_hints

from metroid.utils.validation import check_quantity
from metroid.utils.quantities import extract_spec

P = ParamSpec("P")
R = TypeVar("R")


def enforce_units(func: Callable[P, R]) -> Callable[P, R]:
    """Enforce input and output parameter units of a callback function.

    Parameters
    ----------
    func : callable
        A callback function.

    Returns
    -------
    func: callable
        The callback function after unit validation.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            spec = extract_spec(hints.get(name))
            if spec:
                if value is None:
                    continue
                bound.arguments[name] = check_quantity(value, spec)

        result = func(*bound.args, **bound.kwargs)
        return_spec = extract_spec(hints.get("return"))
        if return_spec and result is not None:
            result = check_quantity(result, return_spec)

        return result

    return wrapper
