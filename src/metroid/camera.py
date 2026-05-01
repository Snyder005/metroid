from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Self

import astropy.units as u
import numpy as np

from metroid.photometry.throughput import ThroughputCurve
from metroid.plugins.discovery import load_entrypoint_plugins
from metroid.plugins.providers import create_provider
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import check_quantity, Gain, PixelScale, QuantumEfficiency, Time
from metroid.utils.validation import get_field_value


class Camera:

    @enforce_units
    def __init__(
        self,
        bandpasses: dict[str, ThroughputCurve],
        gain: Gain,
        pixel_scale: PixelScale,
        qe: QuantumEfficiency = 1.0 * u.electron / u.ph,
    ):
        self._gain = gain
        self._pixel_scale = pixel_scale
        self._qe = qe

        for key, value in bandpasses.items():
            if not isinstance(key, str):
                raise TypeError("must be 'str'")

            if not isinstance(value, ThroughputCurve):
                raise TypeError("must be 'Bandpass'")

        self._bandpasses = MappingProxyType(bandpasses)

    def __getitem__(self, key):
        try:
            return self._bandpasses[key]
        except KeyError:
            raise ValueError(f"unknown bandpass name: {key}") from None

    def __iter__(self):
        return iter(self._bandpasses)

    def __len__(self):
        return len(self._bandpasses)

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
        load_entrypoint_plugins()
        gain = get_field_value(config, "gain", float) * u.electron / u.adu
        pixel_scale = get_field_value(config, "pixel_scale", float) * u.arcsec / u.pix
        qe = get_field_value(config, "qe", float) * u.electron / u.ph
        names = tuple(get_field_value(config, "bands", list))

        bandpasses = {}

        plugin = "rubin"
        provider = create_provider(plugin)
        bandpasses = provider.load(*names)

        return cls(bandpasses, gain, pixel_scale, qe=qe)

    @property
    def filter_names(self) -> tuple[str, ...]:
        """The camera filter bandpass names (`tuple` [`str`], read-only)."""
        return tuple(self._bandpasses.keys())

    @property
    @enforce_units
    def gain(self) -> Gain:
        """The camera gain, in electrons per ADU
        (`astropy.units.Quantity`, read-only).
        """
        return self._gain

    @property
    @enforce_units
    def pixel_scale(self) -> PixelScale:
        """The pixel scale of the camera (`astropy.units.Quantity`,
        read-only).
        """
        return self._pixel_scale

    @property
    @enforce_units
    def qe(self) -> QuantumEfficiency:
        """The quantum efficiency of the camera (`astropy.units.Quantity`,
        read-only).
        """
        return self._qe
