import inspect
from collections.abc import Callable
from functools import wraps
from typing import get_type_hints, ParamSpec, TypeVar

from .quantities import _extract_spec, check_quantity


def enforce_units[**P, R](func: Callable[P, R]) -> Callable[P, R]:
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

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            spec = _extract_spec(hints.get(name))
            if spec and value is not None:
                bound.arguments[name] = check_quantity(value, spec)

        result = func(*bound.args, **bound.kwargs)
        return_spec = _extract_spec(hints.get("return"))
        if return_spec and result is not None:
            result = check_quantity(result, return_spec)

        return result

    return wrapper
