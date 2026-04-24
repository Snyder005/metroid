from astropy import units as u

def flux_to_intensity(flux, distance):
    return (flux * distance**2).to(u.W / u.sr)

def flux_to_surface_brightness(flux, solid_angle):
    return (flux / solid_angle).to(u.W / (u.m**2 * u.sr))

def photon_flux_to_adu():
    ...
