from abc import ABC, abstractmethod
from astropy import units as u
from astropy.constants import G, R_earth, M_earth
import numpy as np
import galsim

from metroid.utils.validation import check_quantity
from metroid.pupils import Pupil

class OrbitalObject(ABC):
    """An abstract base class for orbital objects."""

    def __init__(
        self,
        height: u.Quantity,
        zenith_angle: u.Quantity,
        rotation_angle: u.Quantity = 90 * u.deg,
        nadir_pointing: bool = False,
    ):
        self.height = height
        self.zenith_angle = zenith_angle
        self.rotation_angle = rotation_angle

        if isinstance(nadir_pointing, bool):
            self.nadir_pointing = nadir_pointing
        else:
            raise TypeError("must be 'bool'")

    @property
    def height(self) -> u.Quantity:
        """The orbital height of the object, in kilometers
        (`astropy.units.Quantity`).
        """
        return self._height.to(u.km)

    @height.setter
    def height(self, quantity: u.Quantity) -> None:
        self._height = check_quantity(quantity, u.km, vmin=100.0)

    @property
    def zenith_angle(self) -> u.Quantity:
        """The angle from the telescope zenith to the object, in degrees
        (`astropy.units.Quantity`).
        """
        return self._zenith_angle.to(u.deg)

    @zenith_angle.setter
    def zenith_angle(self, quantity: u.Quantity) -> None:
        self._zenith_angle = check_quantity(quantity, u.deg, vmin=0.0, vmax=90.0)

    @property
    def rotation_angle(self) -> u.Quantity:
        """The rotation angle of the object from the horizon, in degrees
        (`astropy.units.Quantity`)."""
        return self._rotation_angle.to(u.deg)

    @rotation_angle.setter
    def rotation_angle(self, quantity: u.Quantity) -> None:
        self._rotation_angle = check_quantity(quantity, u.deg)

    @property
    def nadir_pointing(self) -> bool:
        return self._nadir_pointing

    @nadir_pointing.setter
    def nadir_pointing(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("must be 'bool'")
        
        self._nadir_pointing = value

    @property
    def nadir_angle(self) -> u.Quantity:
        """The angle from the object nadir to the telescope, in degrees
        (`astropy.units.Quantity`, read-only).
        """
        theta_n = np.arcsin(R_earth * np.sin(self.zenith_angle) / (R_earth + self.height))
        return theta_n.to(u.deg)

    @property
    def distance(self) -> u.Quantity:
        """The distance from the telescope to the object, in kilometers
        (`astropy.units.Quantity`, read-only).
        """
        if np.isclose(self.zenith_angle, 0, atol=1e-09):
            return self.height

        d = np.sin(self.zenith_angle - self.nadir_angle) * R_earth / np.sin(self.nadir_angle)
        return d.to(u.km)

    @property
    def orbital_velocity(self) -> u.Quantity:
        """The orbital velocity of the object, in meters per second
        (`astropy.units.Quantity`, read-only).
        """
        v = np.sqrt(G * M_earth / (R_earth + self.height))
        return v.to(u.m / u.s, equivalencies=u.dimensionless_angles())

    @property
    def orbital_angular_velocity(self) -> u.Quantity:
        """The orbital angular velocity of the object, in radians per second
        (`astropy.units.Quantity`, read-only).
        """
        omega = self.orbital_velocity / (R_earth + self.height)
        return omega.to(u.rad / u.s, equivalencies=u.dimensionless_angles())

    @property
    def perpendicular_velocity(self) -> u.Quantity:
        """The velocity of the object perpendicular to the line-of-sight, in
        meters per second (`astropy.units.Quantity`, read-only).
        """
        v_p = self.orbital_velocity * np.cos(self.nadir_angle)
        return v_p.to(u.m / u.s, equivalencies=u.dimensionless_angles())

    @property
    def perpendicular_angular_velocity(self) -> u.Quantity:
        """The angular velocity of the object perpendicular to the
        line-of-sight, in radians per second (`astropy.units.Quantity`,
        read-only).
        """
        omega_p = self.perpendicular_velocity / self.distance
        return omega_p.to(u.rad / u.s, equivalencies=u.dimensionless_angles())

    @property
    @abstractmethod
    def profile(self) -> galsim.GSObject:
        """The surface brightness profile of the object (`galsim.GSObject`,
        read-only).
        """
        pass

    @property
    @abstractmethod
    def area(self) -> u.Quantity:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        pass

    def calculate_pixel_time(self, pixel_scale: u.Quantity) -> u.Quantity:
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
        t_p = check_quantity(pixel_scale, u.arcsec / u.pix, vmin=0.0) / self.perpendicular_angular_velocity
        return t_p.to(u.s, equivalencies=[(u.pix, None)])

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
            raise TypeError("must be 'metroid.Pupil'")

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

        return profile.rotate(phi).transform(mu, 0.0, 0.0, 1.0).rotate(-phi) / mu


class CircularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a circular disk."""

    def __init__(
        self,
        height: u.Quantity,
        zenith_angle: u.Quantity,
        radius: u.Quantity,
        rotation_angle: u.Quantity = 90 * u.deg,
        nadir_pointing: bool = False,
    ):
        super().__init__(height, zenith_angle, rotation_angle, nadir_pointing)
        self._radius = check_quantity(radius, u.m, vmin=0.0)

    @property
    def radius(self) -> u.Quantity:
        """The radius of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._radius.to(u.m)

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
    def area(self) -> u.Quantity:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        A = np.pi * self.radius**2.0
        return A.to(u.m * u.m)


class RectangularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a rectangle."""

    def __init__(
        self,
        height: u.Quantity,
        zenith_angle: u.Quantity,
        width: u.Quantity,
        length: u.Quantity,
        rotation_angle: u.Quantity = 90 * u.deg,
        nadir_pointing: bool = False,
    ):
        super().__init__(height, zenith_angle, rotation_angle, nadir_pointing)
        self._width = check_quantity(width, u.m, vmin=0.0)
        self._length = check_quantity(length, u.m, vmin=0.0)

    @property
    def width(self) -> u.Quantity:
        """The width of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._width.to(u.m)

    @property
    def length(self) -> u.Quantity:
        """The length of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._length.to(u.m)

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
    def area(self) -> u.Quantity:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        A = self.width * self.length
        return A.to(u.m * u.m)
