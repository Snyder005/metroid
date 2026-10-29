__all__ = ["CircularSurface", "RectangularSurface", "Surface"]

from abc import ABC, abstractmethod

import astropy.units as u
import galsim
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..utils.decorators import enforce_units
from ..utils.quantities import Area, Array, GeometryLength, OrbitalDistance, Reflectivity, Scalar, SolidAngle


class Surface(ABC):
    """An abstract base clase for a single surface."""

    @enforce_units
    def __init__(
        self,
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
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
    @enforce_units
    def reflectivity(self) -> Reflectivity[Scalar]:
        """The reflectivity of the surface, a dimensionless fraction in
        ``[0, 1]`` (`astropy.units.Quantity`, read-only).
        """
        return self._reflectivity

    @enforce_units
    def get_bodyframe_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.GSObject:
        """Get the unprojected body-frame surface brightness profile.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the observer to the body-frame origin, in
            kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The unprojected, flux-scaled, shifted body-frame profile.

        Raises
        ------
        ValueError
            Raised if the surface is edge-on or nearly edge-on.
        """
        distance += self.z0
        profile = self._shape_profile(distance)

        jac = self.orientation.as_matrix()[:2, :2]
        if abs(np.linalg.det(jac)) < 1e-12:
            raise ValueError("The projected affine transform is singular or ill-conditioned.")

        projected = profile.transform(jac[0, 0], jac[0, 1], jac[1, 0], jac[1, 1])

        dx = (self.x0 / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0 / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return projected.shift(dx, dy)

    @enforce_units
    def get_solid_angle(self, distance: OrbitalDistance[Scalar]) -> SolidAngle[Scalar]:
        """The solid angle of the surface, in steradians
        (`astropy.units.Quantity`, read-only).
        """
        return self.area / distance**2

    @abstractmethod
    def _shape_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.GSObject:
        """Construct the unit-flux, un-shifted angular shape at ``distance``.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the observer to the body-frame origin, in
            kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The angular shape profile, before flux scaling and shifting.
        """
        ...
        
class CircularSurface(Surface):
    """A circular surface."""

    @enforce_units
    def __init__(
        self,
        radius: GeometryLength[Scalar],
        centroid: GeometryLength[Array] = np.array([0.0, 0.0, 0.0]) * u.m,
        orientation: Rotation = Rotation.identity(),
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        super().__init__(centroid, orientation, reflectivity)
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

    def _shape_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.TopHat:
        r = (self.radius / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.TopHat(r)


class RectangularSurface(Surface):
    """A rectangular surface."""

    @enforce_units
    def __init__(
        self,
        width: GeometryLength[Scalar],
        length: GeometryLength[Scalar],
        centroid: GeometryLength[Array] = np.array([0.0, 0.0, 0.0]) * u.m,
        orientation: Rotation = Rotation.identity(),
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        super().__init__(centroid, orientation, reflectivity)
        self._width = width
        self._length = length

    @property
    @enforce_units
    def width(self) -> GeometryLength[Scalar]:
        """The width of the surface, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._width

    @property
    @enforce_units
    def length(self) -> GeometryLength[Scalar]:
        """The length of the surface, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._length

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The physical area of the surface, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.width * self.length

    def _shape_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.Box:
        w = (self.width / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.Box(w, l)
