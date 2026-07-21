# utils/

## Files

| File | What | When to read |
| ---- | ---- | ------------ |
| `quantities.py` | `QuantitySpec`, `Spec` builder, `Constraint` protocol, `Range` and `Finite` constraint types, `check_quantity` validator, shape markers (`Scalar`, `Array`, `AnyShape`), `_extract_spec`; catalogue of `QuantitySpec` constants and generic type aliases (`Time`, `Area`, `Gain`, etc.) | Adding a new physical quantity or constraint; debugging unit validation errors; understanding how type aliases are resolved |
| `decorators.py` | `enforce_units` decorator for unit-validating function parameters and return values; `validated_dataclass` wrapper extending enforcement to dataclass `__init__` | Adding `@enforce_units` to a new function or class; debugging decorator behavior; implementing a validated dataclass |
| `validation.py` | `get_field_value` function for typed extraction from configuration dictionaries | Implementing config-driven construction; debugging missing-field or wrong-type errors in config parsing |
| `__init__.py` | Empty package init | - |
