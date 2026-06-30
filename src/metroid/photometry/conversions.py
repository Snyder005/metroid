from .photo_params import PhotometricParameters
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Adu, EnergyFlux, PhotonFlux, Radiance, SolidAngle


@enforce_units
def energy_flux_to_radiance(flux: EnergyFlux, solid_angle: SolidAngle) -> Radiance:
    """Convert an energy flux density to radiance.

    Parameters
    ----------
    flux : `astropy.units.Quantity`
        The energy flux density in units of ergs per second per square meters.
    solid_angle : `astropy.units.Quantity`
        The solid angle of the emitting object.

    Returns
    -------
    radiance : `astropy.units.Quantity`
        The radiance of the emitting object.
    """
    return flux / solid_angle


@enforce_units
def photon_flux_to_adu(photon_flux: PhotonFlux, photo_params: PhotometricParameters) -> Adu:
    """Convert a photon flux density to ADU.

    Parameters
    ----------
    photon_flux : `astropy.units.Quantity`
        The photon flux density in units of photons per second per square
        meters.
    photo_params : `metroid.photometry.PhotometricParameters`
        The photometric parameters of the observation.

    Returns
    -------
    adu : `astropy.units.Quantity`
        The ADU corresponding to the total detected photons.

    Raises
    ------
    TypeError
        Raised if ``photo_params`` is an invalid type.
    """
    if not isinstance(photo_params, PhotometricParameters):
        raise TypeError("photo_params must be 'metroid.photometry.PhotometricParameters'")

    return photon_flux * photo_params.exptime * photo_params.qe * photo_params.area / photo_params.gain
