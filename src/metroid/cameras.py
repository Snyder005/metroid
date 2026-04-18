from astropy import units as u
import copy
import os
import typing
from typing import Protocol, Self

from rubin_sim.phot_utils import Bandpass
from metroid.utils.validation import check_quantity, get_field_value
from metroid.plugins.registry import get_provider


class Camera:

    def __init__(
        self,
        gain: u.Quantity,
        pixel_scale: u.Quantity,
        bandpasses: dict[str, Bandpass],
    ):
        self._gain = check_quantity(gain, u.electron / u.adu, vmin=0.0)
        self._pixel_scale = check_quantity(pixel_scale, u.arcsec / u.pix, vmin=0.0)
        self._bandpasses = {}

        for key, value in bandpasses.items():
            if not isinstance(key, str):
                raise TypeError("must be 'str'")

            if not isinstance(value, Bandpass):
                raise TypeError("must be 'Bandpass'")

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
        gain = get_field_value(config, "gain", float)
        pixel_scale = get_field_value(config, "pixel_scale", float)
        bands = tuple(get_field_value(config, "bands", list))

        bandpasses = {}

        plugin = "rubin"  # for now default is to use rubin_sim
        provider = get_provider(plugin)
        bandpasses = provider.load(*bands)

        return cls(gain * u.electron / u.adu, pixel_scale * u.arcsec / u.pix, bandpasses)

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
        """The camera filter bandpass names (`tuple` [`str`], read-only)."""
        return tuple(self._bandpasses.keys())

    def get_bandpass(self, name: str) -> Bandpass:
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
        return copy.deepcopy(self._bandpasses[name])

    def get_bandpasses(self, *names: str) -> dict[str, Bandpass]:
        """Get a deep copy of the bandpasses or a bandpass subset.

        Parameters
        ----------
        *names : `str`
            Camera filter bandpass names.

        Returns
        -------
        bandpasses : `dict` [`str`, `rubin_sim.phot_utils.Bandpass`]
            A deep copy of the bandpasses or corresponding bandpass subset.
        """
        if not names:
            return copy.deepcopy(self._bandpasses)

        else:
            return {name: copy.deepcopy(self._bandpasses[name]) for name in names}
