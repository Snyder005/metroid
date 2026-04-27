from astropy import units as u

from metroid.photo_params import PhotometricParameters
from metroid.utils.decorators import enforce_units
from metroid.utils.quantities import (
    Adu, 
    EnergyFlux, 
    OrbitalDistance, 
    PhotonFlux, 
    Radiance, 
    RadiantIntensity, 
    SolidAngle
)

@enforce_units
def energy_flux_to_radiant_intensity(flux: EnergyFlux, distance: OrbitalDistance) -> RadiantIntensity:
    return (flux * distance**2).to(u.W / u.sr, equivalencies=u.dimensionless_angles())

@enforce_units
def energy_flux_to_radiance(flux: EnergyFlux, solid_angle: SolidAngle) -> Radiance:
    return (flux / solid_angle).to(u.W / (u.sr * u.m**2))

@enforce_units
def photon_flux_to_adu(photon_flux: PhotonFlux, photo_params: PhotometricParameters) -> Adu:
    if not isinstance(photo_params, PhotometricParameters):
        raise TypeError("must be 'metroid.photo_params.PhotometricParameters'")

    return photon_flux * photo_params.exptime * photo_params.qe * photo_params.area / photo_params.gain
