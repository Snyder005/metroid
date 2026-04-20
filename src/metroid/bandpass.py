import astropy.units as u
import numpy as np
from speclite.filters import Filter

class Bandpass:

    def __init__(self, wavelength, throughput, meta):
        
        self._st = FilterResponse(wavelength, throughput, meta)

    @property
    def wavelength(self):
        return self._st.wavelength # in angstrom

    @property
    def throughput(self):
        return self._st.response # dimensionless unscaled

    def convolve(self, wavelength, flambda):
        wavelength.to(u.AA)
