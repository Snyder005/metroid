__all__ = ["Transform"]

from dataclasses import dataclass
from typing import Self

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Transform:

    rotation: Rotation
    translation: GeometryLength[Array]

    @classmethod
    def identity(cls) -> Self:
        return cls(Rotation.identity(), np.array([0.0, 0.0, 0.0]) * u.m)

    @property
    @enforce_units
    def dx(self) -> GeometryLength[Scalar]:
        """The body-frame x offset of the surface centroid, in meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.translation[0]

    @property
    @enforce_units
    def dy(self) -> GeometryLength[Scalar]:
        """The body-frame y offset of the surface centroid, in meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.translation[1]

    @property
    @enforce_units
    def dz(self) -> GeometryLength[Scalar]:
        """The body-frame z offset of the surface centroid, in meters
        (`astropy.units.Quantity`, read-only).
        """
        return self.translation[2]

    @property
    def normal(self) -> NDArray[np.float32]:
        """The body-frame normal vector of the surface (`numpy.ndarray`,
        read-only).
        """
        return self.rotation.apply([0.0, 0.0, 1.0])
