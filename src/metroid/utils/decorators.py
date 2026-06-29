import inspect
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

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

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            if name in ["self"]:
                continue

            spec, shape = _extract_spec(hints.get(name))
            if spec and value is not None:
                bound.arguments[name] = check_quantity(value, spec, shape)

        result = func(*bound.args, **bound.kwargs)
        return_spec, return_shape = _extract_spec(hints.get("return"))
        if return_spec and result is not None:
            result = check_quantity(result, return_spec, return_shape)

        return result

    return wrapper


def validated_dataclass[T: type](**dc_kwargs: Any) -> Callable[[T], T]:
    """Extended unit enforcement to dataclass types.

    Parameters
    ----------
    dc_kwargs:
        Keyword arguments for the dataclass decorator.

    Returns
    -------
    decorator : decorator
        The modified dataclass decorator with unit enforcement.
    """

    def decorator(cls: T) -> T:
        cls = dataclass(**dc_kwargs)(cls)
        cls.__init__ = enforce_units(cls.__init__)  # type: ignore[attr-defined]
        return cls

    return decorator
