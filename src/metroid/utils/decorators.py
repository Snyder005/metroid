import inspect
from typing import get_type_hints

from metroid.utils.validation import check_quantity

def enforce_units(func):
    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Validate inputs
        for name, value in bound.arguments.items():
            spec = extract_spec(hints.get(name))
            if spec:
                if value is None:
                    continue
                bound.arguments[name] = check_quantity(value, spec)

        # Call function
        result = func(*bound.args, **bound.kwargs)

        # Validate return
        return_spec = extract_spec(hints.get("return"))
        if return_spec and result is not None:
            result = check_quantity(result, return_spec)

        return result

    return wrapper
