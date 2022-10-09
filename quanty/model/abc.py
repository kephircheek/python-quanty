from quanty.geometry import Geometry


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

    def __init__(
        self, geometry: Geometry = None, depth: int = None, h_angle=0, norm_on=(0, 1)
    ):
        self._geometry = geometry
        self._depth = depth
        self._h_angle = h_angle
        self._norm_on = norm_on

    @property
    def geometry(self):
        return self._geometry

    def _constant(self, i, j):
        return 1

    def constant(self, i, j):
        if self._depth is not None and abs(i - j) > self._depth:
            return 0

        if self._norm_on is not None and (i, j) == self._norm_on:
            return 1

        if self._norm_on is None:
            return self._constant(i, j)

        return self._constant(i, j) / self.constant_norm

    @property
    def constant_norm(self):
        raise NotImplementedError
