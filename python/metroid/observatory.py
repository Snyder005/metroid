from abc import ABC, abstractmethod
from astropy import units as u
import galsim

from metroid.utils.validation import check_quantity

class Pupil(ABC):

    @property
    @abstractmethod
    def area(self) -> u.Quantity:
        pass

    @abstractmethod
    def get_profile(self, distance: u.Quantity) -> galsim.GSObject:
        pass

class CircularPupil(Pupil):

    def __init__(self, radius: u.Quantity):
        self._radius = check_quantity(radius, u.m, vmin=0.0)

    @property
    def radius(self) -> u.Quantity:
        return self._radius.to(u.m)

    @property
    def area(self) -> u.Quantity:
        r = self.radius
        
        A = np.pi*r**2
        return A.to(u.m*u.m)

    def get_profile(self, distance: u.Quantity) -> galsim.TopHat:
        distance = check_quantity(distance, u.km, vmin=100.0)
        radius = self.outer_radius/distance

        r = radius.to_value(u.arcsec, equivalences=u.dimensionless_angles())
        profile = galsim.TopHat(r)
        return profile

class AnnularPupil(Pupil):

    def __init__(self, inner_radius: u.Quantity, outer_radius: u.Quantity):
        self._inner_radius = check_quantity(inner_radius, u.m, vmin=0.0)
        self._outer_radius = check_quantity(outer_radius, u.m, vmin=inner_radius.to_value(u.m))
        
    @property
    def inner_radius(self) -> u.Quantity:
        return self._inner_radius.to(u.m)

    @property
    def outer_radius(self) -> u.Quantity:
        return self._outer_radius.to(u.m)

    def get_profile(self, distance: u.Quantity) -> galsim.Sum:
        distance = check_quantity(distance, u.km, vmin=100.0)
        outer_radius = self.outer_radius/distance
        inner_radius = self.inner_radius/distance

        r_o = outer_radius.to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        r_i = inner_radius.to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.0)
        return profile
