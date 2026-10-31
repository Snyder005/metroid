__all__ = ["CircularSurface", "RectangularSurface", "Surface"]

from abc import ABC, abstractmethod

import astropy.units as u
import galsim
import numpy as np

from ..geom.transform import Transform
from ..utils.decorators import enforce_units
from ..typing.aliases import (
    Area,
    Array,
    GeometryLength,
    OrbitalDistance,
    Reflectivity,
    Scalar,
    SolidAngle,
)


class Surface(ABC):
    """An abstract base clase for a single surface."""

    @enforce_units
    def __init__(
        self,
        orientation: Transform = Transform.identity(),
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        self._orientation = orientation
        self._reflectivity = reflectivity

    @property
    @abstractmethod
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The physical area of the surface, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        ...

    @property
    def orientation(self) -> Transform:
        """The surface orientation in the body-frame."""
        return self._orientation

    @orientation.setter
    def orientation(self, transform: Transform) -> None:
        if not isinstance(transform, Transform):
            raise TypeError("must be a 'metroid.geom.Transform'")

        self._orientation = transform

    @property
    @enforce_units
    def reflectivity(self) -> Reflectivity[Scalar]:
        """The reflectivity of the surface, a dimensionless fraction in
        ``[0, 1]`` (`astropy.units.Quantity`, read-only).
        """
        return self._reflectivity

    @abstractmethod
    @enforce_units
    def _profile(
        self,
        distance: OrbitalDistance[Scalar],
    ) -> galsim.GSObject:
        """Get the surface brightness profile of the surface.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the observer to surface, in kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The surface brightness profile.

        Raises
        ------
        ValueError
            Raised if the affine transform is singular or ill-conditioned.
        """
        ...


    @enforce_units
    def get_solid_angle(self, distance: OrbitalDistance[Scalar]) -> SolidAngle[Scalar]:
        """The solid angle of the surface, in steradians
        (`astropy.units.Quantity`, read-only).
        """
        return self.area / distance**2

    @enforce_units
    def get_profile(
        self,
        distance: OrbitalDistance[Scalar],
        other_transform: Transform | None = None,
    ) -> galsim.GSObject:
        """The original surface brightness profile at ``distance``.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the observer to the surface, in kilometers.
        other_transform : `metroid.geom.Transform`, optional
            Additional transform to apply (None, by default).

        Returns
        -------
        profile : `galsim.GSObject`
            The surface brightness profile, before rotation and translation.
        """
        profile = self._profile(distance)

        if other_transform:
            transform = other_transform @ self.orientation
        else:
            transform = self.orientation

        jac = transform.rotation.as_matrix()[:2, :2]
        if abs(np.linalg.det(jac)) < 1e-12:
            raise ValueError("The projected surface is edge-on or nearly edge-on.")

        profile = profile.transform(jac[0, 0], jac[0, 1], jac[1, 0], jac[1, 1])
        translation = (transform.translation / distance).to_value(
            u.arcsec,
            equivalencies=u.dimensionless_angles(),
        )
        profile = profile.shift(translation[0], translation[1])

        return profile


class CircularSurface(Surface):
    """A circular surface."""

    @enforce_units
    def __init__(
        self,
        radius: GeometryLength[Scalar],
        orientation: Transform = Transform.identity(),
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        super().__init__(orientation, reflectivity)
        self._radius = radius

    @property
    @enforce_units
    def radius(self) -> GeometryLength[Scalar]:
        """The radius of the surface, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._radius

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The physical area of the surface, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return np.pi * self.radius**2.0

    @enforce_units
    def _profile(self, distance: OrbitalDistance[Scalar]) -> galsim.TopHat:
        """The original surface brightness profile at ``distance``.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the observer to the surface, in kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The surface brightness profile, before rotation and translation.
        """
        r = (self.radius / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.TopHat(r)


class RectangularSurface(Surface):
    """A rectangular surface."""

    @enforce_units
    def __init__(
        self,
        length: GeometryLength[Scalar],
        width: GeometryLength[Scalar],
        orientation: Transform = Transform.identity(),
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        super().__init__(orientation, reflectivity)
        self._length = length
        self._width = width

    @property
    @enforce_units
    def length(self) -> GeometryLength[Scalar]:
        """The length of the surface, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._length

    @property
    @enforce_units
    def width(self) -> GeometryLength[Scalar]:
        """The width of the surface, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._width

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The physical area of the surface, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.width * self.length

    @enforce_units
    def _profile(self, distance: OrbitalDistance[Scalar]) -> galsim.Box:
        """The original surface brightness profile at ``distance``.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the observer to the surface, in kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The surface brightness profile, before rotation and translation.
        """
        l = (self.length / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        w = (self.width / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.Box(l, w)
