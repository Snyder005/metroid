from __future__ import annotations

from typing import Any

import astropy.units as u
from astropy.coordinates import EarthLocation

from metroid.camera import Camera
from metroid.profiles.pupils import Pupil
from metroid.photometry.photo_params import PhotometricParameters
from metroid.utils.config import load_standard_catalogue
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Time


class Observatory:
    """An astronomical observatory."""

    def __init__(self, camera: Camera, pupil: Pupil, location: EarthLocation):
        if isinstance(camera, Camera):
            self._camera = camera
        else:
            raise ValueError("must be 'Camera'")

        if isinstance(pupil, Pupil):
            self._pupil = pupil
        else:
            raise ValueError("must be 'Pupil'")

        if isinstance(location, EarthLocation):
            self._location = location
        else:
            raise ValueError("must be 'EarthLocation'")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Observatory:
        """Create an `Observatory` from a configuration dictionary.

        The configuration nests one sub-block per component, each delegated to
        the relevant factory.

        Parameters
        ----------
        config : `dict`
            A configuration mapping with fields:

            ``"camera"``
                A `Camera` configuration block (see `Camera.from_config`).
            ``"pupil"``
                A `Pupil` configuration block (see `Pupil.from_config`).
            ``"location"``
                An `EarthLocation` block: either ``{lat, lon, height}`` in
                degrees/degrees/meters, or ``{site}`` naming an astropy site
                (resolved with `EarthLocation.of_site`, which may require
                network access).

        Returns
        -------
        observatory : `Observatory`
            The observatory initialized from the configuration.
        """
        camera = Camera.from_config(config["camera"])
        pupil = Pupil.from_config(config["pupil"])
        location = cls._build_location(config["location"])
        return cls(camera, pupil, location)

    @classmethod
    def from_standard(cls, name: str) -> Observatory:
        """Create an `Observatory` from the bundled catalogue of standards.

        Parameters
        ----------
        name : `str`
            The name/label of a standard object in the bundled catalogue
            (e.g. ``"rubin"``).

        Returns
        -------
        observatory : `Observatory`
            The observatory initialized from the named standard definition.

        Raises
        ------
        ValueError
            Raised if ``name`` is not present in the catalogue.
        """
        catalogue = load_standard_catalogue()

        try:
            config = catalogue[name]
        except KeyError:
            raise ValueError(f"unknown standard object: {name}") from None

        return cls.from_config(config)

    @staticmethod
    def _build_location(config: dict[str, Any]) -> EarthLocation:
        """Build an `EarthLocation` from a configuration block.

        Parameters
        ----------
        config : `dict`
            Either ``{lat, lon, height}`` (degrees, degrees, meters) or
            ``{site}`` naming an astropy site.

        Returns
        -------
        location : `astropy.coordinates.EarthLocation`
            The resolved location.
        """
        if "site" in config:
            return EarthLocation.of_site(config["site"])

        return EarthLocation.from_geodetic(
            lon=config["lon"] * u.deg,
            lat=config["lat"] * u.deg,
            height=config["height"] * u.m,
        )

    @property
    def camera(self) -> Camera:
        """The observatory camera (`metroid.camera.Camera`)."""
        return self._camera

    @property
    def pupil(self) -> Pupil:
        """The observatory telescope pupil (`metroid.pupils.Pupil`)."""
        return self._pupil

    @property
    def location(self) -> EarthLocation:
        """The location of the observatory
        (`astropy.coordinates.EarthLocation`).
        """
        return self._location

    @enforce_units
    def get_photo_params(self, exptime: Time) -> PhotometricParameters:
        """Create photometric parameters for an exposure.

        Parameters
        ----------
        exptime : `astropy.units.Quantity`
            The exposure time.

        Returns
        -------
        photo_params : `metroid.photo_params.PhotometricParameters`
            The photometric parameters for the exposure.

        Raises
        ------
        TypeError
            Raised if ``exptime`` is an invalid type.
        ValueError
            Raised if ``exptime`` has an invalid unit or value.
        """
        photo_params = PhotometricParameters(
            exptime=exptime, gain=self.camera.gain, area=self.pupil.area, qe=self.camera.qe
        )
        return photo_params
