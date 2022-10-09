import numpy as np

from .abc import Model


def dipolar_constant(r, angle):
    return (3 * np.cos(angle) ** 2 - 1) / 2 / r**3


class Homogeneous(Model):
    def _constant(self, i, j):
        r, angle = self._geometry.position(i, j)
        angle -= self._h_angle
        return dipolar_constant(r, angle)

    @property
    def constant_norm(self):
        if self._norm_on is None:
            return 1
        return self._constant(*self._norm_on)
