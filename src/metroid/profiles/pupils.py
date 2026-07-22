from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

import astropy.units as u
import galsim
import numpy as np

from ..utils.validation import get_field_value
from ..utils.decorators import enforce_units
from ..utils.quantities import Area, GeometryLength, OrbitalDistance, Scalar


class Pupil(ABC):
    """Abstract base class for telescope pupils."""

    _registry: ClassVar[dict[str, type[Pupil]]] = {}
    """The registry of Pupil subclasses."""

    def __init_subclass__(cls, pupil_type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if pupil_type:
            cls._registry[pupil_type] = cls

    @classmethod
    def from_config(cls, config: dict[str, str | float]) -> Pupil:
        """Create an instance of a specific subclass of `Pupil` from a
        configuration dictionary.

        This class method defines a standardized way to construct an instance
        of a subclass that is specified in the configuration. It is intended
        for creating an instance from a JSON file that contains a "pupil"
        section.

        Parameters
        ----------
        config : `dict`
            A configuration dictionary of fields each consisting of a name
            (`str`) and value (`str` or `float`). A required field is:

            ``"type"``
                The child class pupil type (`str`).

        Returns
        -------
        pupil : `Pupil`
            An instance of a subclass of `Pupil` initialized with the
            configuration values.

        Raises
        ------
        ValueError
            Raised if a required field does not exist or if the pupil type is
            unknown.
        """
        config = config.copy()
        try:
            pupil_type = config.pop("type")
        except KeyError:
            raise ValueError("config is missing required field 'type'") from None

        try:
            subcls = cls._registry[pupil_type]
        except KeyError:
            raise ValueError(f"unknown pupil type: {pupil_type}") from None

        return subcls._from_config(config)

    @classmethod
    @abstractmethod
    def _from_config(cls, config: dict[str, float]) -> Self:
        """Create an instance of a subclass of `Pupil` from a configuration
        dictionary.

        Parameters
        ----------
        config : `dict` [`str`, `float`]
            A configuration dictionary of fields.

        Returns
        -------
        pupil : `Pupil`
            An instance of a subclass of `Pupil` initialized with the
            configuration.
        """
        pass

    @property
    @abstractmethod
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the pupil, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        pass

    @abstractmethod
    def get_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.GSObject:
        """Get the surface brightness profile of the pupil.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the pupil, in kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The surface brightness profile of the pupil.
        """
        pass


class CircularPupil(Pupil, pupil_type="circular"):
    """A circular telescope pupil."""

    @enforce_units
    def __init__(self, radius: GeometryLength[Scalar]):
        self._radius = radius

    @classmethod
    def _from_config(cls, config: dict[str, float]) -> Self:
        """Create a `CircularPupil` instance from a configuration dictionary.

        Parameters
        ----------
        config : `dict`
            A configuration dictionary with field:

            ``"radius"``
                The radius, in meters (`float`).

        Returns
        -------
        pupil : `Pupil`
            An instance of `CircularPupil` initialized with the configuration.

        Raises
        ------
        ValueError
            Raised if the ``"radius"`` field does not exist.
        TypeError
            Raised if ``radius`` is an invalid type.
        """
        radius = get_field_value(config, "radius", float)
        return cls(radius * u.m)

    @property
    @enforce_units
    def radius(self) -> GeometryLength[Scalar]:
        """The radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._radius

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the pupil, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return np.pi * self.radius**2

    @enforce_units
    def get_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.TopHat:
        """Get the surface brightness profile of the pupil.

        Parameters
        ----------
        distance : `astropy.unit.Quantity`
            The distance from the pupil, in kilometers.

        Returns
        -------
        profile : `galsim.TopHat`
            The surface brightness profile of the pupil.

        Raises
        ------
        TypeError
            Raised if ``distance`` is an invalid type.
        ValueError
            Raised if ``distance`` has an invalid unit or value.
        """
        r = (self.radius / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.TopHat(r)


class AnnularPupil(Pupil, pupil_type="annular"):
    """An annular telescope pupil."""

    @enforce_units
    def __init__(self, inner_radius: GeometryLength[Scalar], outer_radius: GeometryLength[Scalar]):
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius

        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must be greater than inner_radius")

    @classmethod
    def _from_config(cls, config: dict[str, float]) -> Self:
        """Create an AnnularPupil instance from a configuration dictionary.

        Parameters
        ----------
        config : `dict`
            A configuration dictionary with fields:

            ``"inner_radius"``
                The inner radius of the pupil (`float`).
            ``"outer_radius"``
                The outer radius of the pupil (`float`).

        Returns
        -------
        pupil : `Pupil`
            An instance of `AnnularPupil` initialized with the configuration.
        """
        inner_radius = get_field_value(config, "inner_radius", float)
        outer_radius = get_field_value(config, "outer_radius", float)

        return cls(inner_radius * u.m, outer_radius * u.m)

    @property
    @enforce_units
    def inner_radius(self) -> GeometryLength[Scalar]:
        """The inner radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._inner_radius

    @property
    @enforce_units
    def outer_radius(self) -> GeometryLength[Scalar]:
        """The outer radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._outer_radius

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the pupil, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return np.pi * (self.outer_radius**2 - self.inner_radius**2)

    @enforce_units
    def get_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.Sum:
        """Get the surface brightness profile of the pupil.

        Parameters
        ----------
        distance : `astropy.unit.Quantity`
            The distance from the pupil, in kilometers.

        Returns
        -------
        profile : `galsim.Sum`
            The surface brightness profile of the pupil.

        Raises
        ------
        TypeError
            Raised if ``distance`` is an invalid type.
        ValueError
            Raised if ``distance`` has an invalid unit or value.
        """
        r_o = (self.outer_radius / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        r_i = (self.inner_radius / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i / r_o) ** 2)
