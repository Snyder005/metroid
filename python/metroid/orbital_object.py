from abc import ABC, abstractmethod
from astropy import units as u
from astropy.constants as import G, R_earth, M_earth
import numpy as np
import galsim

from metroid.utils.validation import check_quantity
from metroid.observatory import Pupil

class OrbitalObject(ABC):
    """An abstract base class that defines attributes and methods common to
    all orbital objects.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        The orbital height of the object.
    zenith_angle : `astropy.units.Quantity`
        The angle from telescope zenith to the object.
    rotation_angle : `astropy.units.Quantity`, optional
        The rotation angle of the object (90 degrees, by default).
    nadir_pointing : `bool`, optional
        Object is nadir pointing if `True` (`False`, by default).

    Raises
    ------
    TypeError
        Raised if a parameter is an invalid type.
    ValueError
        Raised if a quantity has an invalid unit or value.
    """
    
    nadir_pointing: bool | None = None
    """Nadir-pointing object if `True` (`bool`)."""

    def __init__(
            self,
            height: u.Quantity,
            zenith_angle: u.Quantity,
            rotation_angle: u.Quantity = 90*u.deg,
            nadir_pointing: bool = False,
        ):
        self.height = height
        self.zenith_angle = zenith_angle
        self.rotation_angle = rotation_angle
        if not isinstance(nadir_pointing, bool):
            raise TypeError("must be 'bool'")
        self.nadir_pointing = nadir_pointing

    @property
    def height(self) -> u.Quantity:
        """The orbital height of the object (`astropy.units.Quantity`)."""
        return self._height.to(u.km)

    @height.setter
    def height(self, quantity: u.Quantity) -> None:
        self._height = check_quantity(quantity, u.km, vmin=100.0)

    @property
    def rotation_angle(self) -> u.Quantity:
        """The rotation angle of the object."""
        return self._rotation_angle.to(u.deg)

    @rotation_angle.setter
    def rotation_angle(self, quantity: u.Quantity) -> None:
        self._rotation_angle = check_quantity(quantity, u.deg)

    @property
    def zenith_angle(self) -> u.Quantity:
        """The angle from telescope zenith to the object 
        (`astropy.units.Quantity`).
        """
        return self._zenith_angle.to(u.deg)

    @zenith_angle.setter
    def zenith_angle(self, quantity: u.Quantity) -> None:
        self._zenith_angle = check_quantity(quantity, u.deg, vmin=0.0, vmax=90.0)

    @property
    def nadir_angle(self) -> u.Quantity:
        """The angle from the object nadir to the telescope 
        (`astropy.units.Quantity`, read-only).
        """
        theta_z = self.zenith_angle
        h = self.height

        theta_n = np.arcsin(R_earth*np.sin(theta_z)/(R_earth + h))
        return theta_n.to(u.deg)

    @property
    def distance(self) -> u.Quantity:
        """The distance to the object from the telescope 
        (`astropy.units.Quantity`, read-only).
        """
        theta_n = self.nadir_angle
        theta_z = self.zenith_angle

        if np.isclose(theta_z.value, 0, atol=1e-09):
            return self.height

        d = np.sin(theta_z - theta_n)*R_earth/np.sin(theta_n)
        return d.to(u.km)

    @property
    def orbital_velocity(self) -> u.Quantity:
        """The orbital velocity of the object (`astropy.units.Quantity`, 
        read-only).
        """
        h = self.height

        v = np.sqrt(G*M_earth/(R_earth + h))
        return v.to(u.m/u.s, equivalencies=u.dimensionless_angles())

    @property
    def orbital_angular_velocity(self) -> u.Quantity:
        """The orbital angular velocity of the object 
        (`astropy.units.Quantity`, read-only).
        """
        v_o = self.orbital_velocity
        h = self.height

        omega = v_o/(R_earth + h)
        return omega.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    @property
    def perpendicular_velocity(self) -> u.Quantity:
        """The velocity of the object perpendicular to the line-of-sight 
        vector (`astropy.units.Quantity`, read-only).
        """
        v_o = self.orbital_velocity
        theta_n = self.nadir_angle

        v_p = v_o*np.cos(theta_n)
        return v_p.to(u.m/u.s, equivalencies=u.dimensionless_angles())

    @property
    def perpendicular_angular_velocity(self) -> u.Quantity:
        """Angular velocity of the object perpendicular to the line-of-sight
        vector (`astropy.units.Quantity`, read-only).
        """
        v_p = self.perpendicular_velocity
        d = self.distance

        omega_p = v_p/d
        return omega_p.to(u.rad/u.s, equivalencies=u.dimensionless_angles())

    @property
    @abstractmethod
    def profile(self) -> galsim.GSObject:
        pass

    @property
    @abstractmethod
    def area(self) -> u.Quantity:
        pass

    def calculate_pixel_time(self, pixel_scale: u.Quantity) -> u.Quantity:
        """Calculate the pixel traversal time of the object.

        The pixel traversal time is the time for the object to traverse a 
        single pixel, which depends on the angular velocity of the object
        perpendicular to the line-of-sight vector.

        Parameters
        ----------
        pixel_scale : `astropy.units.Quantity`
            The pixel scale of the imaging device.

        Returns
        -------
        pixel_time : `astropy.units.Quantity`
            The pixel traversal time of the object.

        Raises
        ------
        TypeError
            Raised if `pixel_scale` is an invalid type.
        ValueError
            Raised if `pixel_scale` has an invalid unit or value.
        """
        pixel_scale = check_quantity(pixel_scale, u.arcsec/u.pix, vmin=0.0)
        omega_p = self.perpendicular_angular_velocity

        t_p = pixel_scale/omega_p
        return t_p.to(u.s, equivalencies=[(u.pix, None)])

    def get_tracked_profile(self, psf: galsim.GSObject, telescope_pupil: Pupil) -> galsim.Convolution:
        """Create the tracked profile from the convolution of the atmospheric
        PSF, defocus kernel, and object profiles.

        Parameters
        ----------
        psf : `galsim.GSObject`
            A surface brightness profile representing an atmospheric PSF.
        telescope_pupil: `metroid.Pupil`
            The pupil of the telescope that is observing the object.

        Returns
        -------
        tracked_profile : `galsim.GSObject`
            The tracked profile of the object.

        Raises
        ------
        TypeError
            Raised if `telescope_pupil` or `psf` are invalid types.
        """
        defocus = telescope_pupil.get_profile(self.distance)
        tracked_profile = galsim.Convolve(self.profile, defocus, psf)
        return tracked_profile

    def _project(self, profile: galsim.GSObject) -> galsim.Transformation:
        """Apply angle-of-view projection effects.

        Parameters
        ----------
        profile: `galsim.GSObject`
            A surface brightness profile.

        Returns
        -------
        projected_profile: `galsim.GSObject`
            The projected surface brightness profile.
        """
        mu = np.cos(self.nadir_angle)
        phi = galsim.Angle(self.rotation_angle, galsim.degrees)

        profile = profile.rotate(phi).transform(mu, 0.0, 0.0, 1.0).rotate(-phi)/mu
        return profile

class CircularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a circular disk.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        The orbital height of the object.
    zenith_angle : `astropy.units.Quantity`
        The angle from telescope zenith to the object.
    radius : `astropy.units.Quantity`
        The radius of the object.
    rotation_angle : `astropy.units.Quantity`, optional
        The rotation angle of the object (90 degrees, by default).
    nadir_pointing : `bool`, optional
        Object is nadir pointing if `True` (`False`, by default).

    Raises
    ------
    TypeError
        Raised if a parameter is an invalid type.
    ValueError
        Raised if a quantity has an invalid unit or value.
    """

    def __init__(
            self,
            height: u.Quantity,
            zenith_angle: u.Quantity,
            radius: u.Quantity,
            rotation_angle: u.Quantity = 90*u.km,
            nadir_pointing: bool = False,
        ):
        super().__init__(height, zenith_angle, rotation_angle, nadir_pointing)
        self._radius = check_quantity(radius, u.m, vmin=0.0)

    @property
    def radius(self) -> u.Quantity:
        """The radius of the object in meters (`astropy.units.Quantity`, 
        read-only).
        """
        return self._radius.to(u.m)

    @property
    def profile(self) -> galsim.TopHat:
        """The surface brightness profile of the object (`galsim.TopHat`, 
        read-only).
        """
        r = (self.radius/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())

        profile = galsim.TopHat(r)
        if self.nadir_pointing:
            profile = self._project(profile)
        return profile

    @property
    def area(self) -> u.Quantity:
        """The surface area of the object (`astropy.units.Quantity`, 
        read-only).
        """
        r = self.radius

        A = np.pi*r**2.
        return A.to(u.m*u.m)

class RectangularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a rectangle.

    Parameters
    ----------
    height : `astropy.units.Quantity`
        The orbital height of the object.
    zenith_angle : `astropy.units.Quantity`
        The angle from telescope zenith to the object.
    width : `astropy.units.Quantity`
        The width of the object.
    length : `astropy.units.Quantity`
        The length of the object.
    rotation_angle : `astropy.units.Quantity`, optional
        The rotation angle of the object (90 degrees, by default).
    nadir_pointing : `bool`, optional
        Object is nadir pointing if `True` (`False`, by default).

    Raises
    ------
    TypeError
        Raised if a parameter is an invalid type.
    ValueError
        Raised if a quantity has an invalid unit or value.
    """

    def __init__(
            self,
            height: u.Quantity,
            zenith_angle: u.Quantity,
            width: u.Quantity,
            length: u.Quantity,
            rotation_angle: u.Quantity = 90*u.deg,
            nadir_pointing: bool = False,
        ):
        super().__init__(height, zenith_angle, rotation_angle, nadir_pointing)
        self._width = check_quantity(width)
        self._length = check_quantity(length)

    @property
    def width(self) -> u.Quantity:
        """The width of the object (`astropy.units.Quantity`, read-only).
        """
        return self._width

    @property
    def length(self) -> u.Quantity:
        """The length of the object (`astropy.units.Quantity`, read-only).
        """
        return self._length

    @property
    def profile(self) -> galsim.Box:
        """The surface brightness profile of the object (`galsim.Box`, 
        read-only).
        """
        w = (self.width/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())

        profile = galsim.Box(w, l)
        if self.nadir_pointing:
            profile = self._project(profile)
        return profile
    
    @property
    def area(self) -> u.Quantity:
        """The surface area of the object (`astropy.units.Quantity`, 
        read-only).
        """
        w = self.width
        l = self.length

        A = w*l
        return A.to(u.m*u.m)
