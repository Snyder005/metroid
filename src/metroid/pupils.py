from __future__ import annotations

from abc import ABC, abstractmethod
from astropy import units as u
import galsim
import numpy as np
from typing import ClassVar, Self, Annotated

from metroid.utils.validation import check_quantity, get_field_value
from metroid.utils import quantities as q


class Pupil(ABC):
    """Abstract base class for telescope pupils."""

    _registry: ClassVar[dict[str, type[Pupil]]] = {}

    def __init_subclass__(cls, pupil_type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if pupil_type:
            cls._registry[pupil_type] = cls

    @classmethod
    def from_config(cls, config: dict[str, str | float]) -> Pupil:
        """Create an instance of a specific subclass of `Pupil` from a
        configuration dictionary.

        This class method defines a standardized way to construct an instance
        of a subclass, whose type is specified in the configuration. It is
        intended for creating an instance from a JSON file that contains a
        "pupil" section.

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
            An instance of a subclass of `Pupil` intialized with the
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
            raise ValueError("missing requird field 'type'") from None

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
    def area(self) -> q.Area:
        """The surface area of the pupil, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        pass

    def get_solid_angle(self, distance: q.OrbitalDistance) -> q.SolidAngle:
        """Get the solid angle of the pupil, in steradians.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the pupil, in kilometers.

        Returns
        -------
        solid_angle : `astropy.units.Quantity`
            The solid angle of the pupil, in steradians.

        Raises
        ------
        TypeError
            Raised if ``distance`` is an invalid type.
        ValueError
            Raised if ``distance`` has an invalid unit or value.
        """
        solid_angle = self.area / check_quantity(distance, "orbital_distance") ** 2
        return solid_angle.to(u.sr, equivalencies=u.dimensionless_angles())

    @abstractmethod
    def get_profile(self, distance: q.OrbitalDistance) -> galsim.GSObject:
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

    def __init__(self, radius: q.GeometryLength):
        self._radius = check_quantity(radius, "geometry_length")

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
    def radius(self) -> q.GeometryLength:
        """The radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._radius.to(u.m)

    @property
    def area(self) -> q.Area:
        """The surface area of the pupil, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        r = self.radius
        A = np.pi * r**2

        return A.to(u.m * u.m)

    def get_profile(self, distance: q.OrbitalDistance) -> galsim.TopHat:
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
        distance = check_quantity(distance, "orbital_distance")
        radius = self.radius / distance

        r = radius.to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)

        return profile


class AnnularPupil(Pupil, pupil_type="annular"):

    def __init__(self, inner_radius: q.GeometryLength, outer_radius: q.GeometryLength):
        self._inner_radius = check_quantity(inner_radius, "geometry_length")
        self._outer_radius = check_quantity(outer_radius, "geometry_length")

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
    def inner_radius(self) -> q.GeometryLength:
        """The inner radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._inner_radius.to(u.m)

    @property
    def outer_radius(self) -> q.GeometryLength:
        """The outer radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._outer_radius.to(u.m)

    @property
    def area(self) -> q.Area:
        """The surface area of the pupil, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        r_o = self.outer_radius
        r_i = self.inner_radius
        A = np.pi * (r_o**2 - r_i**2)

        return A.to(u.m * u.m)

    def get_profile(self, distance: q.OrbitalDistance) -> galsim.Sum:
        """Get the surface brightness profile of the pupil

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
        distance = check_quantity(distance, "orbital_distance")
        outer_radius = self.outer_radius / distance
        inner_radius = self.inner_radius / distance

        r_o = outer_radius.to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        r_i = inner_radius.to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i / r_o) ** 2)

        return profile
