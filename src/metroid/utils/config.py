"""Declarative configuration: a format-isolated loader and a class registry.

The only code that knows the on-disk format is YAML lives in `load_yaml`;
everything downstream consumes a plain `dict`. This keeps the format
swappable (a future JSON/TOML loader is additive) and lets tests bypass
files entirely by passing dicts.

`Registrable` generalizes the subclass-registry + `from_config` type
dispatch that `Pupil` originally implemented by hand, so `Camera`,
`Observatory`, and (later) `OrbitalObject` share one implementation with
consistent error semantics.
"""

from __future__ import annotations

from abc import abstractmethod
from importlib import resources
from typing import Any, ClassVar, Self, cast

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    """Parse a YAML file into a dictionary.

    Parameters
    ----------
    path : `str`
        The path to the YAML file.

    Returns
    -------
    config : `dict`
        The parsed configuration mapping.

    Raises
    ------
    TypeError
        Raised if the document does not parse to a mapping.
    """
    with open(path) as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise TypeError("config document must be a mapping")

    return data


def load_standard_catalogue() -> dict[str, Any]:
    """Load the bundled catalogue of standard object definitions.

    The catalogue ships as package data (``metroid/data/standard_objects.yaml``)
    and is resolved with `importlib.resources` so it works from an installed
    package rather than a filesystem-relative path.

    Returns
    -------
    catalogue : `dict`
        A mapping of object name/label to its definition.

    Raises
    ------
    TypeError
        Raised if the catalogue does not parse to a mapping.
    """
    source = resources.files("metroid.data").joinpath("standard_objects.yaml")
    data = yaml.safe_load(source.read_text())

    if not isinstance(data, dict):
        raise TypeError("standard catalogue must be a mapping")

    return data


class Registrable:
    """Mixin providing a subclass registry keyed by a ``type`` string.

    A base class opts in by passing ``registry_label="<noun>"`` to its class
    statement, which gives that base its own fresh ``_registry`` dict.
    Concrete subclasses register themselves by passing ``type="<name>"``.
    `from_config` pops the ``type`` field, looks up the subclass, and delegates
    to its `_from_config`.
    """

    _registry: ClassVar[dict[str, type[Registrable]]]
    """The registry of concrete subclasses, keyed by type string."""

    _registry_label: ClassVar[str]
    """The human-readable noun for this hierarchy, used in error messages."""

    def __init_subclass__(cls, type: str | None = None, registry_label: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # A new hierarchy root gets its own registry so sibling hierarchies
        # (Pupil, Camera, ...) never share a namespace.
        if registry_label is not None:
            cls._registry = {}
            cls._registry_label = registry_label

        if type is not None:
            cls._registry[type] = cls

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Create an instance of the subclass named by the ``type`` field.

        Parameters
        ----------
        config : `dict`
            A configuration mapping. A required field is:

            ``"type"``
                The concrete subclass type (`str`).

        Returns
        -------
        instance
            An instance of the subclass initialized with the configuration.

        Raises
        ------
        ValueError
            Raised if the required ``"type"`` field is missing or if the type
            is unknown.
        """
        config = config.copy()
        label = cls._registry_label

        try:
            type_name = config.pop("type")
        except KeyError:
            raise ValueError("config is missing required field 'type'") from None

        try:
            subcls = cls._registry[type_name]
        except KeyError:
            raise ValueError(f"unknown {label} type: {type_name}") from None

        return cast(Self, subcls._from_config(config))

    @classmethod
    @abstractmethod
    def _from_config(cls, config: dict[str, Any]) -> Self:
        """Create an instance of a concrete subclass from a configuration.

        Parameters
        ----------
        config : `dict`
            A configuration mapping (with the ``"type"`` field already removed).

        Returns
        -------
        instance
            An instance of the subclass initialized with the configuration.
        """
        pass
