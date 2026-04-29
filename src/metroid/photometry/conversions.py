from astropy import units as u

from .photo_params import PhotometricParameters
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import Adu, EnergyFlux, PhotonFlux, Radiance, SolidAngle


@enforce_units
def energy_flux_to_radiance(flux: EnergyFlux, solid_angle: SolidAngle) -> Radiance:
    return flux / solid_angle


@enforce_units
def photon_flux_to_adu(photon_flux: PhotonFlux, photo_params: PhotometricParameters) -> Adu:
    if not isinstance(photo_params, PhotometricParameters):
        raise TypeError("photo_params must be 'metroid.photometry.PhotometricParameters'")

    return photon_flux * photo_params.exptime * photo_params.qe * photo_params.area / photo_params.gain
