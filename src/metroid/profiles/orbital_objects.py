from abc import ABC, abstractmethod

import astropy.units as u
from astropy.constants import G, R_earth, M_earth
import galsim
import numpy as np

from .pupils import Pupil
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import (
    Angle,
    AngularVelocity,
    Area,
    GeometryLength,
    OrbitalDistance,
    PixelScale,
    SolidAngle,
    Time,
    Velocity,
)


class OrbitalObject(ABC):
    """An abstract base class for orbital objects."""

    @enforce_units
    def __init__(
        self,
        height: OrbitalDistance,
        zenith_angle: Angle,
        rotation_angle: Angle = 0.0 * u.deg,
        nadir_pointing: bool = False,
    ):
        self.height = height
        self.zenith_angle = zenith_angle
        self.rotation_angle = rotation_angle
        self.nadir_pointing = nadir_pointing

    @property
    @enforce_units
    def height(self) -> OrbitalDistance:
        """The orbital height of the object, in kilometers
        (`astropy.units.Quantity`).
        """
        return self._height

    @height.setter
    @enforce_units
    def height(self, quantity: OrbitalDistance) -> None:
        self._height = quantity

    @property
    @enforce_units
    def zenith_angle(self) -> Angle:
        """The angle from the telescope zenith to the object, in degrees
        (`astropy.units.Quantity`).
        """
        return self._zenith_angle

    @zenith_angle.setter
    @enforce_units
    def zenith_angle(self, quantity: Angle) -> None:
        self._zenith_angle = quantity

    @property
    @enforce_units
    def rotation_angle(self) -> Angle:
        """The rotation angle of the object from the horizon, in degrees
        (`astropy.units.Quantity`)."""
        return self._rotation_angle

    @rotation_angle.setter
    @enforce_units
    def rotation_angle(self, quantity: Angle) -> None:
        self._rotation_angle = quantity

    @property
    def nadir_pointing(self) -> bool:
        """`True` if object is nadir pointing, `False` otherwise (`bool`)."""
        return self._nadir_pointing

    @nadir_pointing.setter
    def nadir_pointing(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("nadir_pointing must be 'bool'")

        self._nadir_pointing = value

    @property
    @enforce_units
    def nadir_angle(self) -> Angle:
        """The angle from the object nadir to the telescope, in degrees
        (`astropy.units.Quantity`, read-only).
        """
        return np.arcsin(R_earth * np.sin(self.zenith_angle) / (R_earth + self.height))

    @property
    @enforce_units
    def distance(self) -> OrbitalDistance:
        """The distance from the telescope to the object, in kilometers
        (`astropy.units.Quantity`, read-only).
        """
        if np.isclose(self.zenith_angle, 0, atol=1e-09):
            return self.height

        return np.sin(self.zenith_angle - self.nadir_angle) * R_earth / np.sin(self.nadir_angle)

    @property
    @enforce_units
    def orbital_velocity(self) -> Velocity:
        """The orbital velocity of the object, in meters per second
        (`astropy.units.Quantity`, read-only).
        """
        return np.sqrt(G * M_earth / (R_earth + self.height))

    @property
    @enforce_units
    def orbital_angular_velocity(self) -> AngularVelocity:
        """The orbital angular velocity of the object, in radians per second
        (`astropy.units.Quantity`, read-only).
        """
        return self.orbital_velocity / (R_earth + self.height)

    @property
    @enforce_units
    def perpendicular_velocity(self) -> Velocity:
        """The velocity of the object perpendicular to the line-of-sight, in
        meters per second (`astropy.units.Quantity`, read-only).
        """
        v = self.orbital_velocity
        theta = self.nadir_angle
        phi = self.rotation_angle
        return v * np.sqrt(1 - np.sin(theta) ** 2 * np.cos(phi) ** 2)

    @property
    @enforce_units
    def perpendicular_angular_velocity(self) -> AngularVelocity:
        """The angular velocity of the object perpendicular to the
        line-of-sight, in radians per second (`astropy.units.Quantity`,
        read-only).
        """
        return self.perpendicular_velocity / self.distance

    @property
    @enforce_units
    def solid_angle(self) -> SolidAngle:
        """The solid angle of the object, in steradians
        (`astropy.units.Quantity`, read-only).
        """
        return self.area / self.distance**2

    @property
    @abstractmethod
    def profile(self) -> galsim.GSObject:
        """The surface brightness profile of the object (`galsim.GSObject`,
        read-only).
        """
        pass

    @property
    @abstractmethod
    @enforce_units
    def area(self) -> Area:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        pass

    @enforce_units
    def calculate_pixel_time(self, pixel_scale: PixelScale) -> Time:
        """Calculate the pixel traversal time of the object.

        The pixel traversal time is defined as the time it takes for the
        object to move across a single pixel.

        Parameters
        ----------
        pixel_scale : `astropy.units.Quantity`
            The pixel scale of the imaging device, in arcseconds per pixel.

        Returns
        -------
        pixel_time : `astropy.units.Quantity`
            The pixel traversal time of the object, in seconds.

        Raises
        ------
        TypeError
            Raised if ``pixel_scale`` is an invalid type.
        ValueError
            Raised if ``pixel_scale`` has an invalid unit or value.
        """
        return pixel_scale * u.pix / self.perpendicular_angular_velocity

    def get_tracked_profile(self, psf: galsim.GSObject, telescope_pupil: Pupil) -> galsim.Convolution:
        """Get the tracked surface brightness profile of the object.

        Parameters
        ----------
        psf : `galsim.GSObject`
            The surface brightness profile of a point-spread function.
        telescope_pupil: `metroid.Pupil`
            The pupil of the observing telescope.

        Returns
        -------
        tracked_profile : `galsim.Convolution`
            The tracked surface brightness profile of the object.

        Raises
        ------
        TypeError
            Raised if either ``psf`` or ``telescope_pupil`` is an invalid type.
        """
        if not isinstance(telescope_pupil, Pupil):
            raise TypeError("must be 'metroid.profiles.Pupil'")

        if not isinstance(psf, galsim.GSObject):
            raise TypeError("must be 'galsim.GSObject'")

        defocus = telescope_pupil.get_profile(self.distance)
        tracked_profile = galsim.Convolve(self.profile, defocus, psf)
        return tracked_profile

    def _project(self, profile: galsim.GSObject) -> galsim.Transformation:
        """Apply angle-of-view projection effects to a surface brightness
        profile.

        Parameters
        ----------
        profile: `galsim.GSObject`
            The surface brightness profile.

        Returns
        -------
        projected_profile: `galsim.Transformation`
            The transformed surface brightness profile.
        """
        mu = np.cos(self.nadir_angle)
        phi = galsim.Angle(self.rotation_angle, galsim.degrees)

        return profile.rotate(-phi).transform(1.0, 0.0, 0.0, mu).rotate(phi) / mu


class CircularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a circular disk."""

    @enforce_units
    def __init__(
        self,
        height: OrbitalDistance,
        zenith_angle: Angle,
        radius: GeometryLength,
        rotation_angle: Angle = 0.0 * u.deg,
        nadir_pointing: bool = False,
    ):
        super().__init__(height, zenith_angle, rotation_angle, nadir_pointing)
        self._radius = radius

    @property
    @enforce_units
    def radius(self) -> GeometryLength:
        """The radius of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._radius

    @property
    def profile(self) -> galsim.TopHat:
        """The surface brightness profile of the object (`galsim.TopHat`,
        read-only).
        """
        r = (self.radius / self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)

        if self.nadir_pointing:
            return self._project(profile)

        else:
            return profile

    @property
    @enforce_units
    def area(self) -> Area:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return np.pi * self.radius**2.0


class RectangularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a rectangle."""

    def __init__(
        self,
        height: OrbitalDistance,
        zenith_angle: Angle,
        width: GeometryLength,
        length: GeometryLength,
        rotation_angle: Angle = 0.0 * u.deg,
        nadir_pointing: bool = False,
    ):
        super().__init__(height, zenith_angle, rotation_angle, nadir_pointing)
        self._width = width
        self._length = length

    @property
    @enforce_units
    def width(self) -> GeometryLength:
        """The width of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._width

    @property
    @enforce_units
    def length(self) -> GeometryLength:
        """The length of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._length

    @property
    def profile(self) -> galsim.Box:
        """The surface brightness profile of the object (`galsim.Box`,
        read-only).
        """
        w = (self.width / self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length / self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)

        if self.nadir_pointing:
            return self._project(profile)

        else:
            return profile

    @property
    @enforce_units
    def area(self) -> Area:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.width * self.length
