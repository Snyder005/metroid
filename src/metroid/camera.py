from astropy import units as u
import copy
import os
import typing
from typing import Protocol, Self
import numpy as np

from rubin_sim.phot_utils import Bandpass
from metroid.utils.validation import check_quantity, get_field_value
from metroid.plugins.registry import get_provider
from metroid.utils import quantities as q
from metroid.utils.decorators import enforce_units


class Camera:

    @enforce_units
    def __init__(
        self,
        gain: q.Gain,
        pixel_scale: q.PixelScale,
        bandpasses: dict[str, Bandpass],
    ):
        self._gain = gain
        self._pixel_scale = pixel_scale
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
        names = tuple(get_field_value(config, "bands", list))

        bandpasses = {}

        plugin = "rubin"  # for now default is to use rubin_sim
        provider = get_provider(plugin)
        bandpasses = provider.load(*names)

        return cls(gain * u.electron / u.adu, pixel_scale * u.arcsec / u.pix, bandpasses)

    @property
    @enforce_units
    def gain(self) -> q.Gain:
        """The camera gain, in electrons per ADU
        (`astropy.units.Quantity`, read-only).
        """
        return self._gain

    @property
    @enforce_units
    def pixel_scale(self) -> q.PixelScale:
        """The pixel scale of the camera
        (`astropy.units.Quantity`, read-only)."""
        return self._pixel_scale

    @property
    def band_names(self) -> tuple[str, ...]:
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
        try:
            bandpass = self._bandpasses[name]
        except KeyError:
            raise ValueError(f"unknown bandpass name: {name}") from None

        return copy.deepcopy(bandpass)

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
            return {name: self.get_bandpass(name) for name in names}

    def get_throughput(self, name: str) -> float:
        """Get the summed throughput of a bandpass.

        Parameters
        ----------
        name : `str`
            The bandpass name.

        Returns
        -------
        throughput : `float`
            The summed throughput of the bandpass.
        """
        bandpass = self.get_bandpass(name)
        dlambda = bandpass.wavelen[1] - bandpass.wavelen[0]
        throughput = np.sum(bandpass.sb * dlambda / bandpass.wavelen)
        return throughput
