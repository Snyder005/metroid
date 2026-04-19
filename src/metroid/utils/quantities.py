from typing import Annotated
from astropy import units as u

GeometryLength = Annotated[u.Quantity, "geometry_length"]
OrbitalDistance = Annotated[u.Quantity, "orbital_distance"]
Area = Annotated[u.Quantity, "area"]

Time = Annotated[u.Quantity, "time"]
Velocity = Annotated[u.Quantity, "velocity"]

Angle = Annotated[u.Quantity, "angle"]
SolidAngle = Annotated[u.Quantity, "solid_angle"]
AngularVelocity = Annotated[u.Quantity, "angular_velocity"]

ADU = Annotated[u.Quantity, "adu"]
Gain = Annotated[u.Quantity, "gain"]
PixelScale = Annotated[u.Quantity, "pixel_scale"]

RadiantIntensity = Annotated[u.Quantity, "radiant_intensity"]

UNIT_REGISTRY = {
    "geometry_length": {
        "canonical": u.m,
        "allowed": {u.m, u.cm, u.mm},
        "typical_range": (1e-3, 1e3),  # millimeters → km-ish structures
    },
    "orbital_distance": {
        "canonical": u.km,
        "allowed": {u.km, u.m},
        "typical_range": (1e2, 1e6),  # km → Earth orbit scale+
    },
    "area": {
        "canonical": u.m**2,
        "allowed": {u.m**2, u.cm**2, u.mm**2},
        "typical_range": (1e-6, 1e6),
    },
    "time": {
        "canonical": u.s,
        "allowed": {u.s},
    },
    "velocity": {
        "cannonical": u.m / u.s,
        "allowed": {u.m / u.s, u.km / u.s},
    },
    "angle": {
        "canonical": u.deg,
        "allowed": {u.deg, u.rad, u.arcsec},
    },
    "solid_angle": {
        "canonical": u.sr,
        "allowed": {u.sr},
    },
    "angular_velocity": {
        "canonical": u.rad / u.s,
        "allowed": {u.rad / u.s, u.arcsec / u.s, u.deg / u.s},
    },
    "adu": {
        "canonical": u.adu,
        "allowed": {u.adu},
        "typical_range": (1, 1e7),
    },
    "gain": {
        "canonical": u.electron / u.adu,
        "allowed": {u.electron / u.adu},
        "typical_range": (1e-1, 1e2),
    },
    "pixel_scale": {
        "canonical": u.arcsec / u.pix,
        "allowed": {u.arcsec / u.pix},
        "typical_range": (1e-2, 1e1),
    },
    "radiant_intensity": {
        "canonical": u.W / u.sr,
        "allowed": {u.W / u.sr},
    },
}
