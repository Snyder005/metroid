from astropy import units as u
import numpy as np

from metroid.utils import quantity as q

class Sed:

    def __init__(self, wavelength: q.Wavelength, flambda: q.Flux):

        self._wavelength = check_quantity(wavelength, "wavelength")
        self._flambda = check_quantity(flambda, "flux")

    @classmethod
    def for_ab_magnitude(cls, wl_min: float = 300.0, wl_max: float = 1150.0, wl_step: float = 0.1):
        
        wavelength = np.arange(wl_min, wl_max + wl_step, wl_step) * u.AA
        fnu = np.ones(len(wavelength)) * 3631.0 * u.Jy
        flambda = (fnu * c / wavelength ** 2).to(u.erg / (u.s * u.cm ** 2 * u.AA))

        return cls(wavelength, flambda)

    @property
    def wavelength(self):
        return self._wavelength.to(u.AA)

    @property
    def flux(self):
        return self._flux.to(u.erg / (u.s * u.cm ** 2 * u.AA))
