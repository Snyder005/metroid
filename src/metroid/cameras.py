from astropy import units as u
import copy
import os
import typing

from rubin_sim import phot_utils
from rubin_sim.data import get_data_dir
from metroid.utils import check_quantity, get_field_value


class Camera:

    def __init__(
        self,
        pixel_scale: u.Quantity,
        gain: u.Quantity,
        bandpasses: dict[str, phot_utils.Bandpass],
    ):
        self._pixel_scale = check_quantity(pixel_scale, u.arcsec / u.pix, vmin=0.0)
        self._gain = check_quantity(gain, u.electron / u.adu, vmin=0.0)
        self._bandpasses = {}

        for key, value in bandpasses.items():
            if not isinstance(key, str):
                raise TypeError("must be 'str'")

            if not isinstance(value, phot_utils.Bandpass):
                raise TypeError("must be 'phot_utils.Bandpass")

            self._bandpasses[key] = copy.deepcopy(value)

    @classmethod
    def from_config(cls, config: dict[str, str | float]) -> Self:
        """Create a `Camera` instance from a configuration dictionary.

        Parameters
        ----------
        config : `dict`
            A configuration dictionary with fields:

            ``"pixel_scale"``
                The pixel scale of the camera, in arcseconds per pixel
                (`float`).

            ``"gain"``
                The camera gain, in electrons per pixel (`float`).

            ``"bands"``
                A list of camera throughput bands (`list` [`str`]).

        Returns
        -------
        camera : `Camera`
            An instance of `Camera` initialized with the configuration.

        Raises
        ------
        ValueError
            Raised if a required field does not exist.
        TypeError
            Raised if a value is an invalid type.
        """
        pixel_scale = get_field_value(config, "pixel_scale", float)
        gain = get_field_value(config, "gain", float)
        bands = get_field_value(config, "bands", list)

        bandpasses = {}
        for band in bands:
            if not isinstance(band, str):
                raise TypeError("must be 'str'")
            filename = os.path.join(get_data_dir(), "throughputs", "baseline", f"total_{band}.dat")
            bandpass = phot_utils.Bandpass()
            bandpass.read_throughput(filename)
            bandpasses[band] = bandpass

        return cls(pixel_scale * u.arcsec / u.pix, gain * u.electron / u.adu, bandpasses)

    @property
    def gain(self) -> u.Quantity:
        """The camera gain, in electrons per ADU
        (`astropy.units.Quantity`, read-only).
        """
        return self._gain.to(u.electron / u.adu)

    @property
    def pixel_scale(self) -> u.Quantity:
        """The pixel scale of the camera
        (`astropy.units.Quantity`, read-only)."""
        return self._pixel_scale.to(u.arcsec / u.pix)

    @property
    def bands(self) -> tuple[str, ...]:
        """The camera filter bandpasses (`tuple` [`str`], read-only)."""
        return tuple(self._bandpasses.keys())

    def get_bandpass(self, name: str) -> phot_utils.Bandpass:
        """Get a deep copy of a bandpass.

        Parameters
        ----------
        name : `str`
            The bandpass name.

        Returns
        -------
        bandpass : `rubin_sim.phot_utils.Bandpass`
            A deep copy of the corresponding bandpass.
        """
        return copy.deepcopy(self._data[name])
