import astropy.units as u
import galsim

__all__ = ["RectangularComponent", "CircularComponent"]

class BaseComponent:
    """A class representing an orbital object component.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    reflectivity : `float`
        Component reflectivity.
    """
   
    def __init__(self, x0, y0, reflectivity):        
        self._x0 = x0.to(u.m)
        self._y0 = y0.to(u.m)
        self.reflectivity = reflectivity

    @property
    def x0(self):
        """Component centroid position in x-direction 
        (`astropy.units.Quantity`, read-only).
        """
        return self._x0

    @property
    def y0(self):
        """Component centroid position in y-direction 
        (`astropy.units.Quantity`, read-only).
        """
        return self._y0

    @property
    def area(self):
        """Surface area (`astropy.units.Quantity`, read-only).
        """
        return None

    @property
    def flux_scale(self):
        """Flux scale factor (`float`).
        """
        flux_scale = (self.area*self.reflectivity).to_value(u.m*u.m)
        return flux_scale

    def _shift(self, profile, distance):
        """Shift component profile to centroid position.
        """
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)

        return profile
        
class RectangularComponent(BaseComponent):
    """A class representing a rectangular component.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Centroid position of the component in the x-direction.
    y0 : `astropy.units.Quantity`
        Centroid position of the component in the y-direction.
    width : `astropy.units.Quantity`
        Width of the component.
    length : `astropy.units.Quantity`
        Length of the component.
    reflectivity : `float`
        Reflectivity of the component.
    """
        
    def __init__(self, x0, y0, width, length, reflectivity):
        super().__init__(x0, y0, reflectivity)
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        """Width of the component (`astropy.units.Quantity`, read-only).
        """
        return self._width

    @property
    def length(self):
        """Length of the component (`astropy.units.Quantity`, read-only).
        """
        return self._length

    @property
    def area(self):
        """Area of the component (`astropy.units.Quantity`, read-only).
        """
        area = self.width*self.length
        return area

    def create_profile(self, distance, flux=None):
        """Create the component surface brightness profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the composite orbital object.
        flux : `float`, optional
            Number of adu (None, by default).

        Returns
        -------
        profile : `galsim.GSObject`
            Component surface brightness profile.
        """
        w = (self.width/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)
        profile = self._shift(profile, distance)

        if flux is None:
            flux = self.flux_scale
        profile = profile.withFlux(flux)
      
        return profile
       
class Circular(BaseComponent):
    """A class representing a circular component.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Centroid position of the component in the x-direction.
    y0 : `astropy.units.Quantity`
        Centroid position of the component in the y-direction.
    radius : `astropy.units.Quantity`
        Radius of the component.
     reflectivity : `float`
        Reflectivity of the component.
   """
    
    def __init__(self, x0, y0, radius, reflectivity):
        super().__init__(x0, y0, reflectivity)
        self._radius = radius.to(u.m)

    @property
    def radius(self):
        """Radius of the dish (`astropy.units.Quantity`, read-only)."""
        return self._radius
 
    @property
    def area(self):
        """Surface area (`astropy.units.Quantity`, read-only).
        """
        area = np.pi*np.square(self.radius)
        return area
  
    def create_profile(self, distance, flux=None):
        """Create the component surface brightness profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the composite orbital object.
        flux: `float`, optional
            Number of adu. (None, by default).

        Returns
        -------
        profile : `galsim.GSObject`
            Component surface brightness profile.
        """
        r = (self.radius/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)
        profile = self._shift(profile, distance)

        if flux is None:
            flux = self.flux_scale
        profile = profile.withFlux(flux)
       
        return profile
