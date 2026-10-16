"""Tests for metroid.profiles.orbital_objects."""

from astropy import units as u
from astropy.constants import G, M_earth, R_earth
import galsim
import numpy as np
import pytest

from metroid.photometry import PhotometricParameters, Sed, ThroughputCurve
from metroid.profiles.components import CircularComponent, RectangularComponent
from metroid.profiles.orbital_objects import (
    CircularOrbitalObject,
    CompositeOrbitalObject,
    RectangularOrbitalObject,
)
from metroid.profiles.pupils import AnnularPupil, CircularPupil


@pytest.fixture
def bandpass():
    """A ThroughputCurve built from the lsst2023-g filter."""
    return ThroughputCurve.load_filter("lsst2023-g")


@pytest.fixture
def photo_params():
    """Photometric parameters for a representative exposure."""
    return PhotometricParameters(30.0 * u.s, 1.6 * u.electron / u.adu, 35.0 * u.m**2)


@pytest.fixture
def circular_object():
    """A CircularOrbitalObject instance."""
    return CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)


@pytest.fixture
def rectangular_object():
    """A RectangularOrbitalObject instance."""
    return RectangularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 2.0 * u.m, 4.0 * u.m, observed_magnitude=18.0)


@pytest.fixture
def orbital_object(request):
    """An OrbitalObject subclass instance selected by param."""
    if request.param == "circular_object":
        return CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    elif request.param == "rectangular_object":
        return RectangularOrbitalObject(
            550.0 * u.km, 70.0 * u.deg, 2.0 * u.m, 4.0 * u.m, observed_magnitude=18.0
        )
    raise ValueError(f"Unknown object type: {request.param}")


# ---------------------------------------------------------------------------
# Construction and stored attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_stored_attributes(orbital_object):
    """Height, zenith/rotation angle, and pointing angle are stored."""
    assert orbital_object.height == 550.0 * u.km
    assert orbital_object.zenith_angle == 70.0 * u.deg
    assert orbital_object.rotation_angle == 0.0 * u.deg
    assert orbital_object.pointing_angle == 0.0 * u.deg


def test_construction_bad_height_unit():
    """A height with an incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        CircularOrbitalObject(550.0 * u.s, 70.0 * u.deg, 3.0 * u.m)


def test_construction_bad_height_type():
    """A non-Quantity height raises TypeError."""
    with pytest.raises(TypeError):
        CircularOrbitalObject(550.0, 70.0 * u.deg, 3.0 * u.m)


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------


def test_setters_update_values(circular_object):
    """Height, zenith angle, and rotation angle setters update the value."""
    circular_object.height = 600.0 * u.km
    circular_object.zenith_angle = 45.0 * u.deg
    circular_object.rotation_angle = 30.0 * u.deg
    assert circular_object.height == 600.0 * u.km
    assert circular_object.zenith_angle == 45.0 * u.deg
    assert circular_object.rotation_angle == 30.0 * u.deg


def test_height_setter_validates_unit(circular_object):
    """The height setter rejects an incompatible unit."""
    with pytest.raises(ValueError):
        circular_object.height = 600.0 * u.s


def test_pointing_angle_setter_updates_value(circular_object):
    """The pointing_angle setter accepts a value within [0, nadir_angle]."""
    circular_object.pointing_angle = circular_object.nadir_angle
    assert u.isclose(circular_object.pointing_angle, circular_object.nadir_angle)


def test_pointing_angle_setter_rejects_out_of_range(circular_object):
    """The pointing_angle setter rejects values outside nadir_angle +/- 90."""
    with pytest.raises(ValueError):
        circular_object.pointing_angle = circular_object.nadir_angle - 91.0 * u.deg
    with pytest.raises(ValueError):
        circular_object.pointing_angle = circular_object.nadir_angle + 91.0 * u.deg


def test_pointing_angle_setter_rejects_bad_unit(circular_object):
    """The pointing_angle setter rejects an incompatible unit."""
    with pytest.raises(ValueError):
        circular_object.pointing_angle = 1.0 * u.s


def test_construction_rejects_out_of_range_pointing_angle():
    """Constructing with a pointing_angle outside [0, nadir_angle] raises."""
    with pytest.raises(ValueError):
        CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, pointing_angle=200.0 * u.deg)


# ---------------------------------------------------------------------------
# Orbital mechanics (values checked against the analytic formulas)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_orbital_mechanics(orbital_object):
    """Derived orbital quantities match their analytic definitions."""
    h = 550.0 * u.km
    theta_z = 70.0 * u.deg

    theta_n = np.arcsin(R_earth * np.sin(theta_z) / (R_earth + h)).to(u.deg)
    d = (R_earth * np.sin(theta_z - theta_n) / np.sin(theta_n)).to(u.km)
    v_o = np.sqrt(G * M_earth / (R_earth + h)).to(u.m / u.s)
    omega_o = (v_o / (R_earth + h)).to(u.rad / u.s, equivalencies=u.dimensionless_angles())
    v_p = (v_o * np.cos(theta_n)).to(u.m / u.s)
    omega_p = (v_p / d).to(u.rad / u.s, equivalencies=u.dimensionless_angles())
    solid_angle = (orbital_object.area / d**2).to(u.sr, equivalencies=u.dimensionless_angles())

    assert u.isclose(orbital_object.nadir_angle, theta_n)
    assert u.isclose(orbital_object.distance, d)
    assert u.isclose(orbital_object.orbital_velocity, v_o)
    assert u.isclose(orbital_object.orbital_angular_velocity, omega_o)
    assert u.isclose(orbital_object.perpendicular_velocity, v_p)
    assert u.isclose(orbital_object.perpendicular_angular_velocity, omega_p)
    assert u.isclose(orbital_object.solid_angle, solid_angle)


def test_distance_at_zenith_equals_height():
    """At zenith angle 0 the distance equals the orbital height."""
    obj = CircularOrbitalObject(550.0 * u.km, 0.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    assert u.isclose(obj.distance, obj.height)


def test_nadir_angle_units(circular_object):
    """The nadir angle is an angle quantity."""
    assert circular_object.nadir_angle.unit.is_equivalent(u.deg)


# ---------------------------------------------------------------------------
# calculate_pixel_time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_calculate_pixel_time(orbital_object):
    """Pixel traversal time equals pixel_scale / perpendicular angular velocity."""
    pixel_scale = 0.2 * (u.arcsec / u.pix)
    expected = (pixel_scale / orbital_object.perpendicular_angular_velocity).to(
        u.s, equivalencies=[(u.pix, None)]
    )
    assert u.isclose(orbital_object.calculate_pixel_time(pixel_scale), expected)


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_calculate_pixel_time_invalid_unit(orbital_object):
    """A pixel scale with an incompatible unit raises ValueError."""
    with pytest.raises(ValueError):
        orbital_object.calculate_pixel_time(0.2 * (u.s / u.pix))


# ---------------------------------------------------------------------------
# get_tracked_profile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_get_tracked_profile(orbital_object):
    """get_tracked_profile convolves object, defocus, and PSF."""
    psf = galsim.Kolmogorov(fwhm=0.7)
    pupil = CircularPupil(4.0 * u.m)
    assert isinstance(orbital_object.get_tracked_profile(psf, pupil), galsim.Convolution)


def test_get_tracked_profile_bad_pupil(circular_object):
    """A non-Pupil telescope argument raises TypeError."""
    psf = galsim.Kolmogorov(fwhm=0.7)
    with pytest.raises(TypeError):
        circular_object.get_tracked_profile(psf, "not a pupil")


def test_get_tracked_profile_bad_psf(circular_object):
    """A non-GSObject PSF argument raises TypeError."""
    pupil = CircularPupil(4.0 * u.m)
    with pytest.raises(TypeError):
        circular_object.get_tracked_profile("not a psf", pupil)


# ---------------------------------------------------------------------------
# CircularOrbitalObject
# ---------------------------------------------------------------------------


def test_circular_object_radius_and_area(circular_object):
    """The radius property and pi*r^2 area are correct."""
    assert circular_object.radius == 3.0 * u.m
    assert u.isclose(circular_object.area, np.pi * (3.0 * u.m) ** 2)


def test_circular_object_profile_is_projected(circular_object):
    """The profile is a projected GSObject built from a radius/distance TopHat."""
    assert isinstance(circular_object.profile, galsim.GSObject)
    # Projection is not flux-conserving: total flux dims by the projected-area
    # factor mu = cos(nadir_angle - pointing_angle) applied to the unit-flux
    # base shape.
    mu = np.cos(circular_object.nadir_angle - circular_object.pointing_angle).to_value(
        u.dimensionless_unscaled
    )
    assert circular_object.profile.flux == pytest.approx(mu)


# ---------------------------------------------------------------------------
# RectangularOrbitalObject
# ---------------------------------------------------------------------------


def test_rectangular_object_dimensions_and_area(rectangular_object):
    """The width/length properties and w*l area are correct."""
    assert rectangular_object.width == 2.0 * u.m
    assert rectangular_object.length == 4.0 * u.m
    assert u.isclose(rectangular_object.area, (2.0 * u.m) * (4.0 * u.m))


def test_rectangular_object_profile_is_projected(rectangular_object):
    """The profile is a projected GSObject built from a width/length Box."""
    assert isinstance(rectangular_object.profile, galsim.GSObject)
    # Projection is not flux-conserving: total flux dims by the projected-area
    # factor mu applied to the unit-flux base shape.
    mu = np.cos(rectangular_object.nadir_angle - rectangular_object.pointing_angle).to_value(
        u.dimensionless_unscaled
    )
    assert rectangular_object.profile.flux == pytest.approx(mu)


# ---------------------------------------------------------------------------
# Continuous pointing angle projection
# ---------------------------------------------------------------------------


def _projection_extent(profile):
    """The flux-weighted RMS extent of a profile along the projection axis.

    The projection foreshortens along the (unrotated) x-axis by ``mu``, so a
    smaller extent means stronger foreshortening.
    """
    image = profile.drawImage(scale=0.02, method="no_pixel", nx=400, ny=400)
    array = np.clip(image.array, 0.0, None)
    xs = np.mgrid[0 : array.shape[0], 0 : array.shape[1]][1]
    total = array.sum()
    centroid = (array * xs).sum() / total
    return np.sqrt((array * (xs - centroid) ** 2).sum() / total)


def test_observatory_extreme_matches_unprojected():
    """At pointing_angle == nadir_angle the projection is the identity (mu == 1)."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    obj.pointing_angle = obj.nadir_angle

    r = (obj.radius / obj.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
    unprojected = galsim.TopHat(r)

    reference = unprojected.drawImage(scale=0.05, method="no_pixel")
    projected = obj.profile.drawImage(
        scale=0.05, method="no_pixel", nx=reference.array.shape[1], ny=reference.array.shape[0]
    )
    assert np.allclose(projected.array, reference.array, atol=1e-6)


def test_nadir_extreme_is_foreshortened():
    """At the default pointing_angle == 0 the profile is foreshortened (mu < 1)."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)

    r = (obj.radius / obj.distance).to_value(u.arcsec, equivalencies=u.dimensionless_angles())
    unprojected = galsim.TopHat(r)

    assert _projection_extent(obj.profile) < _projection_extent(unprojected)


def test_pointing_angle_monotonic_foreshortening():
    """Foreshortening relaxes monotonically from nadir toward the observatory."""
    nadir = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    nadir_angle = nadir.nadir_angle

    intermediate = CircularOrbitalObject(
        550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, pointing_angle=nadir_angle / 2, observed_magnitude=18.0
    )
    observatory = CircularOrbitalObject(
        550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, pointing_angle=nadir_angle, observed_magnitude=18.0
    )

    extents = [
        _projection_extent(nadir.profile),
        _projection_extent(intermediate.profile),
        _projection_extent(observatory.profile),
    ]
    assert extents[0] < extents[1] < extents[2]


def test_degenerate_geometry_allows_only_zero_pointing_angle():
    """At zenith_angle == 0 the nadir_angle is 0, so only pointing_angle == 0 is valid."""
    obj = CircularOrbitalObject(550.0 * u.km, 0.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    assert u.isclose(obj.nadir_angle, 0.0 * u.deg, atol=1e-12 * u.deg)
    assert obj.pointing_angle == 0.0 * u.deg
    with pytest.raises(ValueError):
        obj.pointing_angle = obj.nadir_angle - 91.0 * u.deg


# ---------------------------------------------------------------------------
# Canonical magnitude
# ---------------------------------------------------------------------------


def test_requires_exactly_one_magnitude():
    """A magnitude is required, given as exactly one of the two kwargs."""
    # Neither given.
    with pytest.raises(ValueError):
        CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m)
    # Both given.
    with pytest.raises(ValueError):
        CircularOrbitalObject(
            550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0, canonical_magnitude=17.0
        )


def test_observed_magnitude_round_trips_at_construction_geometry():
    """observed_magnitude recovers the construction-time observed magnitude."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    assert np.isclose(obj.observed_magnitude, 18.0)


def test_canonical_magnitude_input_is_stored_directly():
    """A canonical_magnitude input is stored as-is, no geometric conversion."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, canonical_magnitude=18.0)
    assert np.isclose(obj.canonical_magnitude, 18.0)
    # Off the canonical geometry the observed magnitude differs from canonical.
    assert not np.isclose(obj.observed_magnitude, 18.0)


def test_canonical_magnitude_differs_from_observed_off_canonical():
    """Away from the canonical geometry the canonical magnitude is shifted."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    # 550 km slant range at 70 deg zenith is farther and foreshortened relative
    # to 500 km at zenith, so the two magnitudes must differ.
    assert not np.isclose(obj.canonical_magnitude, 18.0)


def test_canonical_equals_observed_at_canonical_geometry():
    """At CANONICAL_HEIGHT and zenith, canonical == observed == input."""
    obj = CircularOrbitalObject(500.0 * u.km, 0.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    assert np.isclose(obj.canonical_magnitude, 18.0)
    assert np.isclose(obj.observed_magnitude, 18.0)


def test_canonical_magnitude_invariant_under_geometry_change():
    """canonical_magnitude is invariant while observed_magnitude tracks geometry."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    canonical = obj.canonical_magnitude
    observed = obj.observed_magnitude

    obj.height = 800.0 * u.km
    obj.zenith_angle = 40.0 * u.deg

    assert np.isclose(obj.canonical_magnitude, canonical)
    assert not np.isclose(obj.observed_magnitude, observed)


def test_farther_object_is_fainter_when_observed():
    """A larger canonical->observed distance yields a fainter observed magnitude."""
    near = CircularOrbitalObject(500.0 * u.km, 0.0 * u.deg, 3.0 * u.m, canonical_magnitude=15.0)
    far = CircularOrbitalObject(500.0 * u.km, 0.0 * u.deg, 3.0 * u.m, canonical_magnitude=15.0)
    far.height = 1000.0 * u.km
    assert far.observed_magnitude > near.observed_magnitude


# ---------------------------------------------------------------------------
# Flux scaling (object level)
# ---------------------------------------------------------------------------


def test_calculate_flux_matches_calculate_adu(circular_object, bandpass, photo_params):
    """calculate_flux equals throughput.calculate_adu for a magnitude."""
    assert u.isclose(
        circular_object.calculate_flux(bandpass, photo_params, 18.0),
        bandpass.calculate_adu(18.0, photo_params),
    )


def test_calculate_flux_accepts_sed(circular_object, bandpass, photo_params):
    """calculate_flux accepts an Sed brightness spec."""
    sed = Sed.for_ab_magnitudes()
    assert u.isclose(
        circular_object.calculate_flux(bandpass, photo_params, sed),
        bandpass.calculate_adu(sed, photo_params),
    )


def test_calculate_flux_uses_observed_magnitude(bandpass, photo_params):
    """With no brightness_spec, calculate_flux uses observed_magnitude."""
    obj = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    assert u.isclose(
        obj.calculate_flux(bandpass, photo_params),
        bandpass.calculate_adu(obj.observed_magnitude, photo_params),
    )


def test_calculate_flux_bad_throughput(circular_object, photo_params):
    """A non-ThroughputCurve throughput raises TypeError."""
    with pytest.raises(TypeError):
        circular_object.calculate_flux("not a throughput", photo_params, 18.0)


def test_calculate_flux_bad_photo_params(circular_object, bandpass):
    """A non-PhotometricParameters photo_params raises TypeError."""
    with pytest.raises(TypeError):
        circular_object.calculate_flux(bandpass, "not photo params", 18.0)


@pytest.mark.parametrize("orbital_object", ["circular_object", "rectangular_object"], indirect=True)
def test_get_scaled_profile_flux(orbital_object, bandpass, photo_params):
    """The scaled profile integrates to the ADU total from calculate_flux."""
    profile = orbital_object.get_scaled_profile(bandpass, photo_params, 18.0)
    expected = orbital_object.calculate_flux(bandpass, photo_params, 18.0).to_value(u.adu)
    assert np.isclose(profile.flux, expected)


def test_get_scaled_profile_brighter_is_larger_flux(circular_object, bandpass, photo_params):
    """A brighter (smaller) magnitude yields a larger profile flux."""
    bright = circular_object.get_scaled_profile(bandpass, photo_params, 15.0)
    faint = circular_object.get_scaled_profile(bandpass, photo_params, 20.0)
    assert bright.flux > faint.flux


def test_scaled_profile_flux_independent_of_projection(bandpass, photo_params):
    """The relaxed (non-conserving) projection does not change scaled flux.

    withFlux sets the final total, so the scaled profile integrates to the ADU
    total regardless of the mu-dimming introduced by relaxing _project.
    """
    nadir = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    observatory = CircularOrbitalObject(
        550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, pointing_angle=nadir.nadir_angle, observed_magnitude=18.0
    )
    expected = nadir.calculate_flux(bandpass, photo_params, 18.0).to_value(u.adu)
    assert np.isclose(nadir.get_scaled_profile(bandpass, photo_params, 18.0).flux, expected)
    assert np.isclose(observatory.get_scaled_profile(bandpass, photo_params, 18.0).flux, expected)


def test_get_scaled_tracked_profile_flux(circular_object, bandpass, photo_params):
    """The scaled tracked profile integrates to the ADU total despite a
    non-unit-flux annular defocus (scale applied after convolution)."""
    psf = galsim.Gaussian(sigma=0.5)
    pupil = AnnularPupil(2.5 * u.m, 4.18 * u.m)

    # The defocus profile is deliberately not unit flux.
    assert not np.isclose(pupil.get_profile(circular_object.distance).flux, 1.0)

    tracked = circular_object.get_scaled_tracked_profile(bandpass, photo_params, psf, pupil, 18.0)
    expected = circular_object.calculate_flux(bandpass, photo_params, 18.0).to_value(u.adu)
    assert np.isclose(tracked.flux, expected)


def test_observed_magnitude_to_flux_is_projection_independent(bandpass, photo_params):
    """Total collected flux from an *observed* magnitude carries no projection.

    An observatory measures a tracked satellite's magnitude with no knowledge
    of its pointing/orientation: all flux is assumed collected regardless of
    how the projection spread it. So an observed magnitude must map to the same
    total flux irrespective of pointing_angle -- the projection factor mu must
    NOT enter the observed-magnitude -> flux path. mu enters only the
    canonical<->observed standardization (verified separately below).
    """
    nadir = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    tilted = CircularOrbitalObject(
        550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, pointing_angle=nadir.nadir_angle, observed_magnitude=18.0
    )
    # Same observed magnitude, different projection geometry -> identical flux.
    assert nadir.pointing_angle != tilted.pointing_angle
    assert u.isclose(
        nadir.calculate_flux(bandpass, photo_params),
        tilted.calculate_flux(bandpass, photo_params),
    )


def test_projection_enters_only_canonical_standardization():
    """mu appears only in the canonical<->observed correction, not in flux.

    The correction the software exists to compute is the standardization from
    the geometry-dependent observed magnitude to a geometry-invariant canonical
    magnitude. Two objects with the same observed magnitude but different
    pointing (hence different mu) must therefore have different canonical
    magnitudes -- the projection lives entirely in that conversion.
    """
    nadir = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, observed_magnitude=18.0)
    tilted = CircularOrbitalObject(
        550.0 * u.km, 70.0 * u.deg, 3.0 * u.m, pointing_angle=nadir.nadir_angle, observed_magnitude=18.0
    )
    # Same observed magnitude in, but differing mu -> differing canonical.
    assert np.isclose(nadir.observed_magnitude, tilted.observed_magnitude)
    assert not np.isclose(nadir.canonical_magnitude, tilted.canonical_magnitude)


# ---------------------------------------------------------------------------
# CompositeOrbitalObject
# ---------------------------------------------------------------------------


@pytest.fixture
def composite_object():
    """A CompositeOrbitalObject with a bright bus and a dim panel."""
    bus = RectangularComponent(2.0 * u.m, 2.0 * u.m, reflectivity=0.8 * u.dimensionless_unscaled)
    panel = RectangularComponent(
        1.0 * u.m, 4.0 * u.m, x0=3.0 * u.m, reflectivity=0.3 * u.dimensionless_unscaled
    )
    return CompositeOrbitalObject(550.0 * u.km, 70.0 * u.deg, [bus, panel], observed_magnitude=17.0)


def test_composite_stores_components(composite_object):
    """Components are stored as an immutable tuple."""
    assert isinstance(composite_object.components, tuple)
    assert len(composite_object.components) == 2


def test_composite_rejects_empty_components():
    """An empty component sequence raises ValueError."""
    with pytest.raises(ValueError):
        CompositeOrbitalObject(550.0 * u.km, 70.0 * u.deg, [], observed_magnitude=17.0)


def test_composite_rejects_non_component():
    """A non-Component element raises TypeError."""
    with pytest.raises(TypeError):
        CompositeOrbitalObject(550.0 * u.km, 70.0 * u.deg, ["not a component"], observed_magnitude=17.0)


def test_composite_area_is_sum_of_component_areas(composite_object):
    """The composite area equals the sum of its component areas."""
    expected = sum((c.area for c in composite_object.components), 0.0 * u.m**2)
    assert u.isclose(composite_object.area, expected)


def test_composite_profile_is_projected_gsobject(composite_object):
    """The composite profile is a projected GSObject."""
    assert isinstance(composite_object.profile, galsim.GSObject)


def test_composite_profile_flux_is_sum_of_component_fluxes(composite_object):
    """Composite flux == sum of component relative fluxes dimmed by mu.

    galsim.Sum preserves each summand's flux and the shared projection dims
    each by the same mu = cos(nadir_angle - pointing_angle), so the composite
    flux is mu * sum(relative_flux).
    """
    mu = np.cos(composite_object.nadir_angle - composite_object.pointing_angle).to_value(
        u.dimensionless_unscaled
    )
    expected = mu * sum(c.relative_flux() for c in composite_object.components)
    assert composite_object.profile.flux == pytest.approx(expected)


def test_composite_seam_matches_project_on_sum(composite_object):
    """The per-component projection seam is identical to projecting the sum.

    Projecting each component then summing must equal projecting the summed
    profile once (both _project and galsim.Sum are linear), so the seam is a
    behavior-preserving refactor of the previous project-once-on-the-sum form.
    """
    distance = composite_object.distance
    unprojected_sum = galsim.Sum([c.get_profile(distance) for c in composite_object.components])
    reference = composite_object._project(unprojected_sum)

    seam_image = composite_object.profile.drawImage(scale=0.02, method="no_pixel", nx=256, ny=256)
    reference_image = reference.drawImage(scale=0.02, method="no_pixel", nx=256, ny=256)
    assert np.allclose(seam_image.array, reference_image.array, atol=1e-8)


def test_composite_get_tracked_profile(composite_object):
    """get_tracked_profile is a drop-in with the primitives."""
    psf = galsim.Kolmogorov(fwhm=0.7)
    pupil = CircularPupil(4.0 * u.m)
    assert isinstance(composite_object.get_tracked_profile(psf, pupil), galsim.Convolution)


def test_composite_inherits_orbital_mechanics(composite_object):
    """The composite shares OrbitalObject geometry (distance, nadir angle)."""
    reference = CircularOrbitalObject(550.0 * u.km, 70.0 * u.deg, 1.0 * u.m, observed_magnitude=18.0)
    assert u.isclose(composite_object.distance, reference.distance)
    assert u.isclose(composite_object.nadir_angle, reference.nadir_angle)


# ---------------------------------------------------------------------------
# Composite flux distribution
# ---------------------------------------------------------------------------


def test_composite_scaled_profile_total_flux(composite_object, bandpass, photo_params):
    """The scaled composite integrates to the total ADU from calculate_flux."""
    total_adu = composite_object.calculate_flux(bandpass, photo_params, 17.0).to_value(u.adu)
    scaled = composite_object.get_scaled_profile(bandpass, photo_params, 17.0)
    assert np.isclose(scaled.flux, total_adu)


def test_composite_scaled_profile_distributes_by_relative_weight(composite_object, bandpass, photo_params):
    """The total ADU is distributed across components as total * w_i / sum(w_j).

    galsim.Sum + withFlux scale the whole object linearly, so each component's
    share of the total flux equals its relative-flux weight fraction.
    """
    total_adu = composite_object.calculate_flux(bandpass, photo_params, 17.0).to_value(u.adu)
    weights = [c.relative_flux() for c in composite_object.components]
    weight_sum = sum(weights)

    # The unprojected/projected per-component summands live in profile.obj_list;
    # withFlux scales the whole Sum uniformly, so each component's share of the
    # total ADU is its flux fraction within that Sum.
    summands = composite_object.profile.obj_list
    summed_flux = composite_object.profile.flux
    component_fluxes = [total_adu * obj.flux / summed_flux for obj in summands]

    assert np.isclose(sum(component_fluxes), total_adu)
    for weight, flux in zip(weights, component_fluxes):
        assert np.isclose(flux, total_adu * weight / weight_sum)


def test_composite_scaled_tracked_profile_total_flux(composite_object, bandpass, photo_params):
    """The scaled tracked composite integrates to the total ADU."""
    psf = galsim.Kolmogorov(fwhm=0.7)
    pupil = AnnularPupil(2.5 * u.m, 4.18 * u.m)
    total_adu = composite_object.calculate_flux(bandpass, photo_params, 17.0).to_value(u.adu)
    scaled = composite_object.get_scaled_tracked_profile(bandpass, photo_params, psf, pupil, 17.0)
    assert np.isclose(scaled.flux, total_adu)
