from __future__ import annotations

from types import MappingProxyType
from typing import Any

import astropy.units as u
from speclite.filters import load_filters

from metroid.photometry.throughput import ThroughputCurve
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Gain, PixelScale, QuantumEfficiency
from metroid.utils.validation import get_field_value


class Camera:
    """An imaging camera with named filter bandpasses."""

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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Camera:
        """Create a `Camera` from a configuration dictionary.

        Parameters
        ----------
        config : `dict`
            A configuration mapping with fields:

            ``"bandpasses"``
                The filter set, in one of three forms (see
                `_build_bandpasses`): a list of speclite filter names, a
                single speclite group-wildcard string (e.g. ``"lsst2023-*"``),
                or a mapping of name to inline ``{wavelength, throughput}``
                arrays.
            ``"gain"``
                The gain in electrons per ADU (`float`).
            ``"pixel_scale"``
                The pixel scale in arcseconds per pixel (`float`).
            ``"qe"``
                The quantum efficiency in electrons per photon (`float`,
                optional; defaults to ``1.0``).

        Returns
        -------
        camera : `Camera`
            The camera initialized from the configuration.

        Raises
        ------
        ValueError
            Raised if a required field is missing or bandpass keys collide.
        TypeError
            Raised if a field has an invalid type.
        """
        bandpasses = cls._build_bandpasses(config["bandpasses"])
        gain = get_field_value(config, "gain", float) * (u.electron / u.adu)
        pixel_scale = get_field_value(config, "pixel_scale", float) * (u.arcsec / u.pix)

        if "qe" in config:
            qe = get_field_value(config, "qe", float) * (u.electron / u.ph)
            return cls(bandpasses, gain, pixel_scale, qe)

        return cls(bandpasses, gain, pixel_scale)

    @staticmethod
    def _build_bandpasses(spec: Any) -> dict[str, ThroughputCurve]:
        """Build the bandpass mapping from a configuration specification.

        Three forms are supported:

        - a group-wildcard `str` (e.g. ``"lsst2023-*"``): every filter in the
          speclite group is loaded and keyed by its trailing band letter;
        - a `list` of speclite filter names: each is loaded via
          `ThroughputCurve.load_filter` and keyed by its band letter;
        - a `dict` of name to ``{"wavelength": [...], "throughput": [...]}``:
          each is built as an inline `ThroughputCurve` keyed by that name.

        Parameters
        ----------
        spec : `str`, `list`, or `dict`
            The bandpass specification.

        Returns
        -------
        bandpasses : `dict` [`str`, `ThroughputCurve`]
            The named bandpasses.

        Raises
        ------
        ValueError
            Raised if two bandpasses resolve to the same key.
        TypeError
            Raised if ``spec`` is an unsupported type.
        """
        bandpasses: dict[str, ThroughputCurve] = {}

        def add(key: str, curve: ThroughputCurve) -> None:
            if key in bandpasses:
                raise ValueError(f"duplicate bandpass key: {key}")
            bandpasses[key] = curve

        if isinstance(spec, str):
            for fr in load_filters(spec):
                add(fr.name.split("-")[-1], ThroughputCurve.from_filter_response(fr))

        elif isinstance(spec, list):
            for name in spec:
                add(name.split("-")[-1], ThroughputCurve.load_filter(name))

        elif isinstance(spec, dict):
            for key, arrays in spec.items():
                wavelength = arrays["wavelength"] * u.AA
                throughput = arrays["throughput"] * u.dimensionless_unscaled
                add(key, ThroughputCurve(wavelength, throughput, {"group_name": key, "band_name": key}))

        else:
            raise TypeError("bandpasses must be a 'str', 'list', or 'dict'")

        return bandpasses

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
