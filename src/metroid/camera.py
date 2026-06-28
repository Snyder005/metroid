from __future__ import annotations

from types import MappingProxyType

import astropy.units as u

from metroid.photometry.throughput import ThroughputCurve
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Gain, PixelScale, QuantumEfficiency


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
