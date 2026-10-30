__all__ = ["OrbitalObject"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import galsim

from .surfaces import Surface
from ..geom.transform import Transform
from ..typing.aliases import OrbitalDistance, Scalar
from ..utils.decorators import enforce_units


class OrbitalObject:
    """An abstract base class for orbital objects."""

    @enforce_units
    def __init__(
        self,
        components: Sequence[Surface],
        height: OrbitalDistance[Scalar],
        attitude: Transform = Transform.identity(),
    ):
        components = tuple(components)
        if not components:
            raise ValueError("components must be non-empty")

        for component in components:
            if not isinstance(component, Surface):
                raise TypeError("must be 'metroid.profiles.Surface'")

        self._components = components
        self._height = height
        self._attitude = attitude

    @property
    def components(self) -> tuple[Surface, ...]:
        """The parts making up the composite object
        (`tuple` [`metroid.profiles.Component`, ...], read-only).
        """
        return self._components

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
    def attitude(self) -> Transform:
        return self._attitude

    @enforce_units
    def get_profile(
        self,
        distance: OrbitalDistance[Scalar],
        orbit: Transform = None,
    ) -> galsim.Sum:
        """The surface brightness profile of the object, in the body-frame.


        Revisit this seam -- promoting the shared ``self._project(...)`` to a
        per-component projection that computes each component's own ``mu`` from
        its own body-frame normal and the shared viewing geometry -- when any
        of the following becomes true: (1) components gain independent
        orientation/pointing (their own normals), (2) a 3-D body is projected
        to a 2-D observed profile, or (3) per-component projected-area flux
        weighting is required (e.g. a deployed solar panel seen edge-on). Until
        then the shared angle keeps behavior identical to projecting the sum.
        """
        if orbit:
            transform = orbit @ self.attitude
        else:
            transform = self.attitude
        profiles = [component.get_profile(distance, transform) for component in self.components]

        return galsim.Sum(profiles)
