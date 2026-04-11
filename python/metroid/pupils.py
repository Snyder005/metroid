from __future__ import annotations

from abc import ABC, abstractmethod
from astropy import units as u
import galsim
from typing import Self, ClassVar

from metroid.utils.validation import check_quantity

class Pupil(ABC):
    """Abstract base class for telescope pupils."""
    _registry = ClassVar[dict[str, type[Pupil]]] = {}

    def __init_subclass__(cls, pupil_type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if pupil_type:
            cls._registry[pupil_type] = cls

    @classmethod
    def from_config(cls, config: dict[str, str | float]) -> Pupil:
        config = config.copy()

        try:
            pupil_type = config.pop('type')
        except KeyError:
            raise ValueError("missing requird field 'type'") from None

        try:
            subcls = cls._registry[pupil_type]
        except KeyError:
            raise ValueError(f"unknown pupil type: {pupil_type}") from None

        return subcls._from_config(config)

    @classmethod
    def from_json(cls, infile: str) -> Pupil:

        with open(infile) as f:
            full_config = json.load(f)

        try:
            config = full_config['pupil'] #
        except KeyError:
            raise ValueError("JSON must contain 'pupil' section")

        return cls.from_config(config)

    @classmethod
    @abstractmethod
    def _from_config(cls, config: dict[str, float]) -> Self:
        pass

    @property
    @abstractmethod
    def area(self) -> u.Quantity:
        """Surface area, in square meters (`astropy.units.Quantity`, 
        read-only).
        """
        pass

    @abstractmethod
    def get_profile(self, distance: u.Quantity) -> galsim.GSObject:
        """Get the surface brightness profile.

        Parameters
        ----------
        distance : `astropy.units.Quantity`
            Distance from the pupil, in kilometers.

        Returns
        -------
        profile : `galsim.GSObject`
            Surface brightness profile.

        Raises
        ------
        TypeError
            Raised if ``distance`` is an invalid type.
        ValueError
            Raised if ``distance`` has an invalid unit or value.
        """
        pass

class CircularPupil(Pupil):
    """Circular telescope pupil."""

    def __init__(self, radius: u.Quantity):
        self._radius = check_quantity(radius, u.m, vmin=0.0)

    @classmethod
    def _from_config(cls, config: dict[str, float]) -> Self:

        try:
            radius = config['radius']
        except KeyError:
            raise ValueError("missing required field 'radius'")

        if not isinstance(radius, float):
            raise TypeError("must be 'float'")

        return cls(radius*u.m)

    @property
    def radius(self) -> u.Quantity:
        """Radius, in meters (`astropy.units.Quantity`, read-only)."""
        return self._radius.to(u.m)

    @property
    def area(self) -> u.Quantity:
        """Surface area, in square meters (`astropy.units.Quantity`, 
        read-only).
        """
        r = self.radius
        
        A = np.pi*r**2
        return A.to(u.m*u.m)

    def get_profile(self, distance: u.Quantity) -> galsim.TopHat:
        """Get the surface brightness profile.

        Parameters
        ----------
        distance : `astropy.unit.Quantity`
            Distance from the pupil, in kilometers.

        Returns
        -------
        profile : `galsim.TopHat`
            Radial tophat profile.

        Raises
        ------
        TypeError
            Raised if ``distance`` is an invalid type.
        ValueError
            Raised if ``distance`` has an invalid unit or value.
        """
        distance = check_quantity(distance, u.km, vmin=100.0)
        radius = self.outer_radius/distance

        r = radius.to_value(u.arcsec, equivalences=u.dimensionless_angles())
        profile = galsim.TopHat(r)
        return profile

class AnnularPupil(Pupil):

    def __init__(self, inner_radius: u.Quantity, outer_radius: u.Quantity):
        self._inner_radius = check_quantity(inner_radius, u.m, vmin=0.0)
        self._outer_radius = check_quantity(outer_radius, u.m, vmin=inner_radius.to_value(u.m))
        
    @classmethod
    def _from_config(cls, config: dict[str, float]) -> Self:

        try:
            inner_radius = config['inner_radius']
        except KeyError:
            raise ValueError(f"missing required field 'inner_radius'")

        if not isinstance(inner_radius, float):
            raise TypeError("must be 'float'")

        try:
            outer_radius = config['outer_radius']
        except KeyError:
            raise ValueError(f"missing required field 'outer_radius'")

        if not isinstance(inner_radius, float):
            raise TypeError("must be 'float'")
        
        return cls(inner_radius*u.m, outer_radius*u.m)

    @property
    def inner_radius(self) -> u.Quantity:
        """Inner radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._inner_radius.to(u.m)

    @property
    def outer_radius(self) -> u.Quantity:
        """Outer radius of the pupil, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._outer_radius.to(u.m)

    @property
    def area(self) -> u.Quantity:
        """Surface area, in square meters (`astropy.units.Quantity`,
        read-only).
        """
        r_o = self.outer_radius
        r_i = self.inner_radius

        A = np.pi*(r_o**2 - r_i**2)
        return A.to(u.m*u.m)

    def get_profile(self, distance: u.Quantity) -> galsim.Sum:
         """Get the surface brightness profile.

        Parameters
        ----------
        distance : `astropy.unit.Quantity`
            Distance from the pupil, in units of km.

        Returns
        -------
        profile : `galsim.Sum`
            Annular profile.

        Raises
        ------
        TypeError
            Raised if ``distance`` is an invalid type.
        ValueError
            Raised if ``distance`` has an invalid unit or value.
        """
        distance = check_quantity(distance, u.km, vmin=100.0)
        outer_radius = self.outer_radius/distance
        inner_radius = self.inner_radius/distance

        r_o = outer_radius.to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        r_i = inner_radius.to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r_o) - galsim.TopHat(r_i, flux=(r_i/r_o)**2.0)
        return profile
