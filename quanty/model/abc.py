from dataclasses import dataclass
from quanty.geometry import Geometry


@dataclass(frozen=True)
class Model:
    """
    Model upon geometry.

    Parameters
    ----------
    geometry: Geometry
        Default model is a equivalent particle model with single constant that equals one.
    depth: int
        Number of nearest neighbours levels to interact.
    h_angle: float
        Angle between geometry main axis and external magnetic field.
    norm_on: tuple[int, int]
        Indices of geometry elements which interaction constant equal to one.
    """

    geometry: Geometry = None
    depth: int = None
    h_angle: float = 0
    norm_on: tuple[int, int] = (0, 1)

    def __post_init__(self):
        object.__setattr__(self, "norm_on", tuple(self.norm_on))

    def _constant(self, i, j):
        return 1

    def constant(self, i, j):
        if self.depth is not None and abs(i - j) > self.depth:
            return 0

        if self.norm_on is not None and (i, j) == self.norm_on:
            return 1

        if self.norm_on is None:
            return self._constant(i, j)

        return self._constant(i, j) / self.constant_norm

    @property
    def constant_norm(self):
        raise NotImplementedError
