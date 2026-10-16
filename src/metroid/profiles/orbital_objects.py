from abc import ABC, abstractmethod
from collections.abc import Sequence

from astropy.constants import G, R_earth, M_earth
import astropy.units as u
import galsim
import numpy as np

from .components import Component
from .pupils import Pupil
from ..photometry.photo_params import PhotometricParameters
from ..photometry.sed import Sed
from ..photometry.throughput import ThroughputCurve
from ..utils.decorators import enforce_units
from ..utils.quantities import (
    Adu,
    Angle,
    AngularVelocity,
    Area,
    GeometryLength,
    OrbitalDistance,
    PixelScale,
    Scalar,
    SolidAngle,
    Time,
    Velocity,
)

CANONICAL_RANGE: OrbitalDistance[Scalar] = 500.0 * u.km
"""The canonical reference orbital range for the standardized magnitude.

The canonical geometry is observed face-on (``projection_angle = 0``) where
the line-of-sight distance equals the range and the projection factor ``mu``
is 1. A construction-time observed magnitude is converted to the brightness
the object would have at this reference geometry and stored, because this is
invariant under changes to the (mutable) orbital geometry.
"""


class OrbitalObject(ABC):
    """An abstract base class for orbital objects."""

    @enforce_units
    def __init__(
        self,
        height: OrbitalDistance[Scalar],
        zenith_angle: Angle[Scalar],
        rotation_angle: Angle[Scalar] = 0.0 * u.deg,
        pointing_angle: Angle[Scalar] = 0.0 * u.deg,
        *,
        observed_magnitude: float | None = None,
        canonical_magnitude: float | None = None,
    ):
        self.height = height
        self.zenith_angle = zenith_angle
        self.rotation_angle = rotation_angle
        # Assign last: the pointing_angle setter validates against nadir_angle,
        # which is derived from height and zenith_angle.
        self.pointing_angle = pointing_angle

        ## Modify below, only require observed magnitude, no canonical
        # A magnitude is required, given as exactly one of the two measurables
        # an observatory reports: the observed magnitude at this object's
        # construction geometry, or a standardized "average"/canonical
        # magnitude. Accepting both would let a caller supply a pair that
        # violates their fixed geometric relationship, so exactly one is
        # allowed.
        if (observed_magnitude is None) == (canonical_magnitude is None):
            raise ValueError(
                "exactly one of observed_magnitude or canonical_magnitude is required (not both)"
            )

        ## This likely gets modified, to preserve the unscaled output flux
        # Store the geometry-invariant canonical magnitude, not the observed
        # one: height/zenith_angle/pointing_angle are mutable, so a stored
        # observed magnitude would go stale when the geometry changes. An
        # observed input is converted to canonical at the construction geometry.
        if canonical_magnitude is not None:
            self._canonical_magnitude: float = float(canonical_magnitude)
        else:
            assert observed_magnitude is not None  # guaranteed by the check above
            self._canonical_magnitude = self._observed_to_canonical(float(observed_magnitude))

    @property
    @enforce_units
    def height(self) -> OrbitalDistance[Scalar]:
        """The orbital height of the object, in kilometers
        (`astropy.units.Quantity`).
        """
        return self._height

    @height.setter
    @enforce_units
    def height(self, quantity: OrbitalDistance[Scalar]) -> None:
        self._height = quantity

    @property
    @enforce_units
    def zenith_angle(self) -> Angle[Scalar]:
        """The angle from the telescope zenith to the object, in degrees
        (`astropy.units.Quantity`).
        """
        return self._zenith_angle

    @zenith_angle.setter
    @enforce_units
    def zenith_angle(self, quantity: Angle[Scalar]) -> None:
        self._zenith_angle = quantity

    @property
    @enforce_units
    def rotation_angle(self) -> Angle[Scalar]:
        """The rotation angle of the object from the horizon, in degrees
        (`astropy.units.Quantity`).
        """
        return self._rotation_angle

    @rotation_angle.setter
    @enforce_units
    def rotation_angle(self, quantity: Angle[Scalar]) -> None:
        self._rotation_angle = quantity

    @property
    @enforce_units
    def pointing_angle(self) -> Angle[Scalar]:
        """The pointing angle of the object, measured from its nadir direction
        toward the telescope line of sight, in degrees
        (`astropy.units.Quantity`).
        """
        return self._pointing_angle

    @pointing_angle.setter
    @enforce_units
    def pointing_angle(self, quantity: Angle[Scalar]) -> None:
        nadir_angle = self.nadir_angle
        if not (nadir_angle - 90 * u.deg) < quantity < (nadir_angle + 90 * u.deg):
            raise ValueError(
                f"pointing_angle must be within {nadir_angle.to(u.deg)} +/- 90 deg, got {quantity.to(u.deg)}"
            )

        self._pointing_angle = quantity

    @property
    @enforce_units
    def nadir_angle(self) -> Angle[Scalar]:
        """The angle from the object nadir to the telescope, in degrees
        (`astropy.units.Quantity`, read-only).
        """
        return np.arcsin(R_earth * np.sin(self.zenith_angle) / (R_earth + self.height))

    @property
    @enforce_units
    def distance(self) -> OrbitalDistance[Scalar]:
        """The distance from the telescope to the object, in kilometers
        (`astropy.units.Quantity`, read-only).
        """
        if np.isclose(self.zenith_angle, 0, atol=1e-09):
            return self.height

        return np.sin(self.zenith_angle - self.nadir_angle) * R_earth / np.sin(self.nadir_angle)

    @property
    @enforce_units
    def orbital_velocity(self) -> Velocity[Scalar]:
        """The orbital velocity of the object, in meters per second
        (`astropy.units.Quantity`, read-only).
        """
        return np.sqrt(G * M_earth / (R_earth + self.height))

    @property
    @enforce_units
    def orbital_angular_velocity(self) -> AngularVelocity[Scalar]:
        """The orbital angular velocity of the object, in radians per second
        (`astropy.units.Quantity`, read-only).
        """
        return self.orbital_velocity / (R_earth + self.height)

    @property
    @enforce_units
    def perpendicular_velocity(self) -> Velocity[Scalar]:
        """The velocity of the object perpendicular to the line-of-sight, in
        meters per second (`astropy.units.Quantity`, read-only).
        """
        v = self.orbital_velocity
        theta = self.nadir_angle
        phi = self.rotation_angle
        return v * np.sqrt(1 - np.sin(theta) ** 2 * np.cos(phi) ** 2)

    @property
    @enforce_units
    def perpendicular_angular_velocity(self) -> AngularVelocity[Scalar]:
        """The angular velocity of the object perpendicular to the
        line-of-sight, in radians per second (`astropy.units.Quantity`,
        read-only).
        """
        return self.perpendicular_velocity / self.distance

    @property
    @enforce_units
    def solid_angle(self) -> SolidAngle[Scalar]:
        """The solid angle of the object, in steradians
        (`astropy.units.Quantity`, read-only).
        """
        return self.area / self.distance**2

    @property
    def canonical_magnitude(self) -> float:
        """The standardized AB magnitude at the canonical reference geometry
        (`float`, read-only).

        The canonical geometry is `CANONICAL_RANGE` observed face-on. This
        value is invariant under changes to the object's orbital geometry.
        """
        return self._canonical_magnitude

    @property
    def observed_magnitude(self) -> float:
        """The AB magnitude at the object's *current* geometry (`float`,
        read-only).

        Re-derived from `canonical_magnitude` for the present
        `height`/`zenith_angle`/`pointing_angle`.
        """
        return self._canonical_to_observed(self._canonical_magnitude)

    def _observed_to_canonical(self, magnitude: float) -> float:
        """Convert an observed AB magnitude to the canonical-geometry magnitude.

        The conversion is a flux ratio and therefore purely geometric (and
        band-independent): reflected flux scales as ``mu / distance**2``
        (projection foreshortening times inverse-square range), so
        ``m_canonical = m_observed + 2.5 * log10(mu) - 5 * log10(d / d_can)``.

        Parameters
        ----------
        magnitude : `float`
            The observed AB magnitude at the current geometry.

        Returns
        -------
        canonical_magnitude : `float`
            The AB magnitude at the canonical reference geometry.
        """
        mu = np.cos(self.nadir_angle - self.pointing_angle).to_value(u.dimensionless_unscaled)
        distance_ratio = (self.distance / CANONICAL_RANGE).to_value(u.dimensionless_unscaled)
        return magnitude + 2.5 * np.log10(mu) - 5.0 * np.log10(distance_ratio)

    def _canonical_to_observed(self, magnitude: float) -> float:
        """Convert a canonical-geometry AB magnitude to the observed magnitude.

        The inverse of `_observed_to_canonical` for the current geometry:
        ``m_observed = m_canonical - 2.5 * log10(mu) + 5 * log10(d / d_can)``.

        Parameters
        ----------
        magnitude : `float`
            The AB magnitude at the canonical reference geometry.

        Returns
        -------
        observed_magnitude : `float`
            The observed AB magnitude at the current geometry.
        """
        mu = np.cos(self.nadir_angle - self.pointing_angle).to_value(u.dimensionless_unscaled)
        distance_ratio = (self.distance / CANONICAL_RANGE).to_value(u.dimensionless_unscaled)
        return magnitude - 2.5 * np.log10(mu) + 5.0 * np.log10(distance_ratio)

    @property
    @abstractmethod
    def profile(self) -> galsim.GSObject:
        """The surface brightness profile of the object (`galsim.GSObject`,
        read-only).
        """
        pass

    @property
    @abstractmethod
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        pass

    @enforce_units
    def calculate_pixel_time(self, pixel_scale: PixelScale[Scalar]) -> Time[Scalar]:
        """Calculate the pixel traversal time of the object.

        The pixel traversal time is defined as the time it takes for the
        object to move across a single pixel.

        Parameters
        ----------
        pixel_scale : `astropy.units.Quantity`
            The pixel scale of the imaging device, in arcseconds per pixel.

        Returns
        -------
        pixel_time : `astropy.units.Quantity`
            The pixel traversal time of the object, in seconds.

        Raises
        ------
        TypeError
            Raised if ``pixel_scale`` is an invalid type.
        ValueError
            Raised if ``pixel_scale`` has an invalid unit or value.
        """
        return pixel_scale * u.pix / self.perpendicular_angular_velocity

    def get_tracked_profile(self, psf: galsim.GSObject, telescope_pupil: Pupil) -> galsim.Convolution:
        """Get the tracked surface brightness profile of the object.

        Parameters
        ----------
        psf : `galsim.GSObject`
            The surface brightness profile of a point-spread function.
        telescope_pupil: `metroid.Pupil`
            The pupil of the observing telescope.

        Returns
        -------
        tracked_profile : `galsim.Convolution`
            The tracked surface brightness profile of the object.

        Raises
        ------
        TypeError
            Raised if either ``psf`` or ``telescope_pupil`` is an invalid type.
        """
        if not isinstance(telescope_pupil, Pupil):
            raise TypeError("must be 'metroid.profiles.Pupil'")

        if not isinstance(psf, galsim.GSObject):
            raise TypeError("must be 'galsim.GSObject'")

        defocus = telescope_pupil.get_profile(self.distance)
        tracked_profile = galsim.Convolve(self.profile, defocus, psf)
        return tracked_profile

    @enforce_units
    def calculate_flux(
        self,
        throughput: ThroughputCurve,
        photo_params: PhotometricParameters,
        brightness_spec: float | int | Sed | None = None,
    ) -> Adu[Scalar]:
        """Calculate the total ADU flux the object's profile should carry.

        The magnitude→ADU bridge: the observatory-facing brightness is an AB
        magnitude, which is routed through the photometry layer to a total
        ADU. This is the single place the photometry dependency enters
        `OrbitalObject`; the scaled-profile methods build on it.

        Parameters
        ----------
        throughput : `metroid.photometry.ThroughputCurve`
            The bandpass through which the object is observed.
        photo_params : `metroid.photometry.PhotometricParameters`
            The photometric parameters of the observation.
        brightness_spec : `float`, `int`, `metroid.photometry.Sed`, or `None`
            The brightness specification: an AB magnitude or an object SED. If
            `None` (the default), the object's `observed_magnitude` is used.

        Returns
        -------
        adu : `astropy.units.Quantity`
            The summed ADU of the observation.

        Raises
        ------
        TypeError
            Raised if ``throughput`` or ``photo_params`` is an invalid type.
        """
        if not isinstance(throughput, ThroughputCurve):
            raise TypeError("throughput must be 'metroid.photometry.ThroughputCurve'")

        if not isinstance(photo_params, PhotometricParameters):
            raise TypeError("photo_params must be 'metroid.photometry.PhotometricParameters'")

        if brightness_spec is None:
            brightness_spec = self.observed_magnitude

        return throughput.calculate_adu(brightness_spec, photo_params)

    def get_scaled_profile(
        self,
        throughput: ThroughputCurve,
        photo_params: PhotometricParameters,
        brightness_spec: float | int | Sed | None = None,
    ) -> galsim.GSObject:
        """Get the object's profile scaled to its absolute ADU flux.

        The bare (untracked) profile, for studying an object's surface
        brightness without a PSF or pupil, rescaled so that it integrates to
        the total ADU from `calculate_flux`.

        Parameters
        ----------
        throughput : `metroid.photometry.ThroughputCurve`
            The bandpass through which the object is observed.
        photo_params : `metroid.photometry.PhotometricParameters`
            The photometric parameters of the observation.
        brightness_spec : `float`, `int`, `metroid.photometry.Sed`, or `None`
            The brightness specification (see `calculate_flux`).

        Returns
        -------
        profile : `galsim.GSObject`
            The profile scaled to the absolute ADU flux.
        """
        total_adu = self.calculate_flux(throughput, photo_params, brightness_spec)
        return self.profile.withFlux(total_adu.to_value(u.adu))

    def get_scaled_tracked_profile(
        self,
        throughput: ThroughputCurve,
        photo_params: PhotometricParameters,
        psf: galsim.GSObject,
        telescope_pupil: Pupil,
        brightness_spec: float | int | Sed | None = None,
    ) -> galsim.GSObject:
        """Get the tracked profile scaled to its absolute ADU flux.

        The full convolved profile (object, pupil defocus, PSF) rescaled so
        that it integrates to the total ADU from `calculate_flux`. The flux is
        applied *after* convolution: the pupil defocus profile is not
        unit-flux (e.g. an annular pupil carries flux ``1 - (r_i/r_o)**2``) and
        `galsim.Convolve` multiplies summand fluxes, so `withFlux` on the
        convolved result normalizes to unit flux and rescales to exactly the
        target ADU regardless of intermediate flux bookkeeping.

        Parameters
        ----------
        throughput : `metroid.photometry.ThroughputCurve`
            The bandpass through which the object is observed.
        photo_params : `metroid.photometry.PhotometricParameters`
            The photometric parameters of the observation.
        psf : `galsim.GSObject`
            The surface brightness profile of a point-spread function.
        telescope_pupil : `metroid.Pupil`
            The pupil of the observing telescope.
        brightness_spec : `float`, `int`, `metroid.photometry.Sed`, or `None`
            The brightness specification (see `calculate_flux`).

        Returns
        -------
        tracked_profile : `galsim.GSObject`
            The tracked profile scaled to the absolute ADU flux.

        Raises
        ------
        TypeError
            Raised if ``psf`` or ``telescope_pupil`` is an invalid type.
        """
        total_adu = self.calculate_flux(throughput, photo_params, brightness_spec)
        tracked_profile = self.get_tracked_profile(psf, telescope_pupil)
        return tracked_profile.withFlux(total_adu.to_value(u.adu))

    def _project(self, profile: galsim.GSObject) -> galsim.Transformation:
        """Apply angle-of-view projection effects to a surface brightness
        profile.

        The foreshortening is not flux-conserving: ``transform(mu, 0, 0, 1)``
        scales total flux by the Jacobian ``mu = cos(nadir_angle -
        pointing_angle)``, and that scaling is deliberately kept (there is no
        compensating ``/ mu``). A diffuse (Lambertian) surface seen off-normal
        reflects less total light toward the observer in proportion to its
        projected area, so total flux dims by ``mu``. For an absolutely-scaled
        profile this is invisible because `get_scaled_profile` /
        `get_scaled_tracked_profile` set the final total with `withFlux`; it
        matters for the bare `profile` flux and for future per-component
        projected-area weighting.

        Parameters
        ----------
        profile: `galsim.GSObject`
            The surface brightness profile.

        Returns
        -------
        projected_profile: `galsim.Transformation`
            The transformed surface brightness profile.
        """
        mu = np.cos(self.nadir_angle - self.pointing_angle)
        phi = galsim.Angle(self.rotation_angle.to_value(u.deg), unit=galsim.degrees)

        return profile.rotate(phi).transform(mu, 0.0, 0.0, 1.0).rotate(-phi)


class CircularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a circular disk."""

    @enforce_units
    def __init__(
        self,
        height: OrbitalDistance[Scalar],
        zenith_angle: Angle[Scalar],
        radius: GeometryLength[Scalar],
        rotation_angle: Angle[Scalar] = 0.0 * u.deg,
        pointing_angle: Angle[Scalar] = 0.0 * u.deg,
        *,
        observed_magnitude: float | None = None,
        canonical_magnitude: float | None = None,
    ):
        super().__init__(
            height,
            zenith_angle,
            rotation_angle,
            pointing_angle,
            observed_magnitude=observed_magnitude,
            canonical_magnitude=canonical_magnitude,
        )
        self._radius = radius

    @property
    @enforce_units
    def radius(self) -> GeometryLength[Scalar]:
        """The radius of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._radius

    @property
    def profile(self) -> galsim.Transformation:
        """The surface brightness profile of the object, foreshortened by the
        pointing-angle projection (`galsim.Transformation`, read-only).
        """
        r = (self.radius / self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.TopHat(r)

        return self._project(profile)

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return np.pi * self.radius**2.0


class RectangularOrbitalObject(OrbitalObject):
    """An orbital object in the shape of a rectangle."""

    @enforce_units
    def __init__(
        self,
        height: OrbitalDistance[Scalar],
        zenith_angle: Angle[Scalar],
        width: GeometryLength[Scalar],
        length: GeometryLength[Scalar],
        rotation_angle: Angle[Scalar] = 0.0 * u.deg,
        pointing_angle: Angle[Scalar] = 0.0 * u.deg,
        *,
        observed_magnitude: float | None = None,
        canonical_magnitude: float | None = None,
    ):
        super().__init__(
            height,
            zenith_angle,
            rotation_angle,
            pointing_angle,
            observed_magnitude=observed_magnitude,
            canonical_magnitude=canonical_magnitude,
        )
        self._width = width
        self._length = length

    @property
    @enforce_units
    def width(self) -> GeometryLength[Scalar]:
        """The width of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._width

    @property
    @enforce_units
    def length(self) -> GeometryLength[Scalar]:
        """The length of the object, in meters (`astropy.units.Quantity`,
        read-only).
        """
        return self._length

    @property
    def profile(self) -> galsim.Transformation:
        """The surface brightness profile of the object, foreshortened by the
        pointing-angle projection (`galsim.Transformation`, read-only).
        """
        w = (self.width / self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        l = (self.length / self.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
        profile = galsim.Box(w, l)

        return self._project(profile)

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the object, in square meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.width * self.length


class CompositeOrbitalObject(OrbitalObject):
    """An orbital object assembled from multiple `Component` parts.

    A composite shares a single orbit for the whole rigid body; each component
    contributes an unprojected body-frame profile, which is projected along the
    shared line of sight and then summed.
    """

    @enforce_units
    def __init__(
        self,
        height: OrbitalDistance[Scalar],
        zenith_angle: Angle[Scalar],
        components: Sequence[Component],
        rotation_angle: Angle[Scalar] = 0.0 * u.deg,
        pointing_angle: Angle[Scalar] = 0.0 * u.deg,
        *,
        observed_magnitude: float | None = None,
        canonical_magnitude: float | None = None,
    ):
        super().__init__(
            height,
            zenith_angle,
            rotation_angle,
            pointing_angle,
            observed_magnitude=observed_magnitude,
            canonical_magnitude=canonical_magnitude,
        )

        components = tuple(components)
        if not components:
            raise ValueError("components must be non-empty")

        for component in components:
            if not isinstance(component, Component):
                raise TypeError("must be 'metroid.profiles.Component'")

        self._components = components

    @property
    def components(self) -> tuple[Component, ...]:
        """The parts making up the composite object
        (`tuple` [`metroid.profiles.Component`, ...], read-only).
        """
        return self._components

    @property
    def profile(self) -> galsim.Sum:
        """The surface brightness profile of the object, foreshortened by the
        pointing-angle projection (`galsim.Sum`, read-only).

        Each component profile is built at the composite's `distance`,
        projected, and then summed with `galsim.Sum`. Projecting each component
        before summing is a *seam* for future work: with a single shared
        projection angle it is provably identical to projecting the summed
        profile once (both `_project` and `galsim.Sum` are linear about the
        body-frame origin), so it changes nothing today.

        Revisit this seam -- promoting the shared ``self._project(...)`` to a
        per-component projection that computes each component's own ``mu`` from
        its own body-frame normal and the shared viewing geometry -- when any
        of the following becomes true: (1) components gain independent
        orientation/pointing (their own normals), (2) a 3-D body is projected
        to a 2-D observed profile, or (3) per-component projected-area flux
        weighting is required (e.g. a deployed solar panel seen edge-on). Until
        then the shared angle keeps behavior identical to projecting the sum.
        """
        profiles = [self._project(component.get_profile(self.distance)) for component in self.components]

        return galsim.Sum(profiles)

    @property
    @enforce_units
    def area(self) -> Area[Scalar]:
        """The surface area of the object, the sum of its component areas, in
        square meters (`astropy.units.Quantity`, read-only).
        """
        return sum((component.area for component in self.components), 0.0 * u.m**2)
