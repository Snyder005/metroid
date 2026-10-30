from __future__ import annotations

__all__ = ["Transform"]

from dataclasses import dataclass
from typing import Self

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..utils.decorators import enforce_units, validated_dataclass
from ..typing.aliases import Angle, Array, GeometryLength, Scalar

_NULL_VECTOR = np.zeros(3) * u.m


@validated_dataclass(frozen=True)
class Transform:

    rotation: Rotation
    translation: GeometryLength[Array]

    def __matmul__(self, other: Transform):
        return Transform(
            self.rotation * other.rotation,
            self.rotation.apply(other.translation.value) * other.translation.unit + self.translation,
        )

    @classmethod
    def identity(cls) -> Self:
        return cls(Rotation.identity(), _NULL_VECTOR)

    @classmethod
    @enforce_units
    def from_euler(
        cls,
        axis: str,
        angle: Angle,
        translation: GeometryLength[Array] = _NULL_VECTOR,
    ) -> Self:
        rotation = Rotation.from_euler(axis, angle.value, degrees=True)
        return cls(rotation, translation)

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


def compose(a: Transform, b: Transform) -> Transform:
    """Compose two transforms: a @ b."""
    return a @ b
