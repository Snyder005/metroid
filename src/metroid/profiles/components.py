from abc import ABC, abstractmethod

import astropy.units as u
import galsim
import numpy as np

from ..utils.decorators import enforce_units
from ..utils.quantities import Area, GeometryLength, OrbitalDistance, Reflectivity, Scalar


class Component(ABC):
    """An abstract base class for a single part of a composite satellite.

    A component describes one piece of a satellite (a bus, a solar panel, a
    dish) in the satellite's own *body frame*: a shape, a local centroid
    offset ``(x0, y0)`` in meters, and a reflectivity. It deliberately holds
    **no** orbital state; all orbital mechanics (distance, velocity,
    projection) belong to the enclosing `CompositeOrbitalObject`, which shares
    one orbit for the whole rigid body. This mirrors how `Pupil` is
    orbit-agnostic and receives ``distance`` at ``get_profile`` time.
    """

    @enforce_units
    def __init__(
        self,
        x0: GeometryLength[Scalar] = 0.0 * u.m,
        y0: GeometryLength[Scalar] = 0.0 * u.m,
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        self._x0 = x0
        self._y0 = y0
        self._reflectivity = reflectivity

    @property
    @enforce_units
    def x0(self) -> GeometryLength[Scalar]:
        """The body-frame x offset of the component centroid, in meters
        (`astropy.units.Quantity`, read-only).
        """
        return self._x0

    @property
    @enforce_units
    def y0(self) -> GeometryLength[Scalar]:
        """The body-frame y offset of the component centroid, in meters
        (`astropy.units.Quantity`, read-only).
        """
        return self._y0

    @property
    @enforce_units
    def reflectivity(self) -> Reflectivity[Scalar]:
        """The reflectivity of the component, a dimensionless fraction in
        ``[0, 1]`` (`astropy.units.Quantity`, read-only).
        """
        return self._reflectivity

    @property
    @abstractmethod
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The physical surface area of the component, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        pass

    def relative_flux(self) -> float:
        """The relative surface-brightness weight of the component.

        Under a fully diffuse (Lambertian) assumption a component's reflected
        surface brightness (radiance) is proportional to its reflectivity, so
        the total reflected signal is proportional to ``reflectivity * area``.
        That product is used as the per-component galsim flux, so a
        `galsim.Sum` of components yields physically correct *relative*
        brightness between parts. This is a relative model only; absolute
        photometric normalization is deferred to the flux-scaling roadmap
        (issue #35). Keep this computation isolated so #35 can extend it.

        Returns
        -------
        flux : `float`
            The relative flux weight, ``reflectivity * area`` in square meters.
        """
        return (self.reflectivity * self.area).to_value(u.m**2)

    @enforce_units
    def get_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.GSObject:
        """Get the unprojected body-frame surface brightness profile.

        The physical shape is converted to an angular size at ``distance``,
        scaled to the component's `relative_flux`, and shifted by the angular
        body-frame offset. The returned profile is **unprojected**: only
        `CompositeOrbitalObject.profile` applies the line-of-sight projection,
        once, to the summed profile.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the telescope to the satellite, in kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The unprojected, flux-scaled, shifted body-frame profile.

        Raises
        ------
        TypeError
            Raised if ``distance`` is an invalid type.
        ValueError
            Raised if ``distance`` has an invalid unit or value.
        """
        profile = self._shape_profile(distance).withFlux(self.relative_flux())
        dx = (self.x0 / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0 / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return profile.shift(dx, dy)

    @abstractmethod
    def _shape_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.GSObject:
        """Build the unit-flux, un-shifted angular shape at ``distance``.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            The distance from the telescope to the satellite, in kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            The angular shape profile, before flux scaling and shifting.
        """
        pass


class CircularComponent(Component):
    """A circular disk component (e.g. a dish)."""

    @enforce_units
    def __init__(
        self,
        radius: GeometryLength[Scalar],
        x0: GeometryLength[Scalar] = 0.0 * u.m,
        y0: GeometryLength[Scalar] = 0.0 * u.m,
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        super().__init__(x0, y0, reflectivity)
        self._radius = radius

    @property
    @enforce_units
    def radius(self) -> GeometryLength[Scalar]:
        """The radius of the component, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._radius

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the component, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return np.pi * self.radius**2.0

    def _shape_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.TopHat:
        r = (self.radius / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.TopHat(r)


class RectangularComponent(Component):
    """A rectangular component (e.g. a bus or a solar panel)."""

    @enforce_units
    def __init__(
        self,
        width: GeometryLength[Scalar],
        length: GeometryLength[Scalar],
        x0: GeometryLength[Scalar] = 0.0 * u.m,
        y0: GeometryLength[Scalar] = 0.0 * u.m,
        reflectivity: Reflectivity[Scalar] = 1.0 * u.dimensionless_unscaled,
    ):
        super().__init__(x0, y0, reflectivity)
        self._width = width
        self._length = length

    @property
    @enforce_units
    def width(self) -> GeometryLength[Scalar]:
        """The width of the component, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._width

    @property
    @enforce_units
    def length(self) -> GeometryLength[Scalar]:
        """The length of the component, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._length

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the component, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.width * self.length

    def _shape_profile(self, distance: OrbitalDistance[Scalar]) -> galsim.Box:
        w = (self.width / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length / distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        return galsim.Box(w, l)
