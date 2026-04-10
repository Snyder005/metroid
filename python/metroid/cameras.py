# To Do:
# * Add init from JSON
# * Add init from site name -> JSON file
from astropy import units as u
import copy
import typing

from rubin_sim import phot_utils
from metroid.utils import check_quantity

class Camera:

    def __init__(
            self,
            pixel_scale: u.Quantity,
            gain: u.Quantity,
            bandpasses: dict[str, phot_utils.Bandpass],
        ):
        self._pixel_scale = check_quantity(pixel_scale, u.arcsec/u.pix, vmin=0.0)
        self._gain = check_quantity(gain, u.electron/u.adu, vmin=0.0)

        self._bandpasses = {}
        for key, value in bandpasses.items():
            if not isinstance(key, str):
                raise TypeError("must be 'str'")
            if not isinstance(value, phot_utils.Bandpass):
                raise TypeError("must be 'phot_utils.Bandpass")
            self._bandpasses[key] = copy.deepcopy(value)

    @classmethod
    def from_site(cls, site_name: str) -> Self:
        pass

    @property
    def gain(self) -> u.Quantity:
        return self._gain.to(u.electron/u.adu)

    @property
    def pixel_scale(self) -> u.Quantity:
        return self._pixel_scale.to(u.arcsec/u.pix)

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(self._bandpasses.keys())

    def get_bandpass(self, name: str) -> phot_utils.Bandpass:
        return copy.deepcopy(self._data[name])
