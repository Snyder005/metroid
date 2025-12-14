import astropy.units as u
import galsim

__all__ = ["Panel", "Bus", "Dish"]

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
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    width : `astropy.units.Quantity`
        Width of the solar panel.
    length : `astropy.units.Quantity`
        Length of the solar panel.
    reflectivity : `float`
        Component reflectivity.
    """
        
    def __init__(self, x0, y0, width, length, reflectivity):
        super().__init__(x0, y0, reflectivity)
        
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        """Width of the solar panel (`astropy.units.Quantity`, read-only)."""
        return self._width

    @property
    def length(self):
        """Length of the solar panel (`astropy.units.Quantity`, read-only)."""
        return self._length        

    def create_profile(self, distance):
        """Create solar panel profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the satellite.

        Returns
        -------
        profile : `galsim.GSObject`
            Solar panel profile.
        """
        w = (self.width/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
    
class Bus(Component):
    """A class representing a satellite bus.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    width : `astropy.units.Quantity`
        Width of the bus.
    length : `astropy.units.Quantity`
        Length of the bus.
    flux : `float`, optional
        Total flux of component in electrons (1.0, by default).
    """
        
    def __init__(self, x0, y0, width, length, flux=1.0):
        super().__init__(x0, y0, flux=flux)
        
        self._width = width.to(u.m)
        self._length = length.to(u.m)

    @property
    def width(self):
        """Width of the bus (`astropy.units.Quantity`, read-only)."""
        return self._width

    @property
    def length(self):
        """Length of the bus (`astropy.units.Quantity`, read-only)."""
        return self._length
        
    def create_profile(self, distance):
        """Create bus profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the satellite.

        Returns
        -------
        profile : `galsim.GSObject`
            Bus profile.
        """
        w = (self.width/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
    
class Dish(Component):
    """A class representing a satellite dish.

    Parameters
    ----------
    x0 : `astropy.units.Quantity`
        Component centroid position in x-direction.
    y0 : `astropy.units.Quantity`
        Component centroid position in y-direction.
    radius : `astropy.units.Quantity`
        Radius of the dish.
    flux : `float`, optional
        Total flux of component in electrons (1.0, by default).
    """
    
    def __init__(self, x0, y0, radius, flux=1.0):
        super().__init__(x0, y0, flux=flux)
        
        self._radius = radius

    @property
    def radius(self):
        """Radius of the dish (`astropy.units.Quantity`, read-only)."""
        return self._radius
        
    def create_profile(self, distance):
        """Create dish profile.
    
        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance to the satellite.

        Returns
        -------
        profile : `galsim.GSObject`
            Dish profile.
        """
        r = (self.radius/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)
        profile = profile.withFlux(self.flux)
        
        dx = (self.x0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        dy = (self.y0/distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = profile.shift(dx, dy)
        
        return profile
