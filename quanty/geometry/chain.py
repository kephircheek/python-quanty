from typing import List

import numpy as np

from .abc import Geometry


class Chain(Geometry):
    def position(self, i, j):
        """Return distance and angle between chain direction and connection line."""
        raise NotImplementedError


class UniformChain(Chain):
    def __init__(self, r=1):
        self._r = r

    def position(self, i, j):
        return abs(i - j) * self._r, 0


class ZigZagChain(Chain):
    @classmethod
    def from_two_chain(cls, r, offset, width):
        raise NotImplementedError

    def __init__(self, angle, ra=1, rb=None):
        """
        Parameters
        -----------
        angle:
            Angle between first connection line and chain direction.
        ra:
            Distance between first two nodes in chain.
        rb: optional
            Distance between first two nodes in chain.
            Default equal to `ra`.

        """
        if angle >= np.pi / 2 or angle < -np.pi / 2:
            raise ValueError(f"angle should be in semi-interval [0, pi/2), not {angle}")
        self._angle = angle
        self._ra = ra
        self._rb = rb

    @property
    def ra(self):
        """Returns distance between nearest odd and even chain nodes."""
        return self._ra

    @property
    def rb(self):
        return self._ra if self.is_isosceles else self._rb

    @property
    def is_isosceles(self):
        return self._rb is None

    @property
    def rc(self):
        """Returns distance between nearest same parity chain nodes."""
        if self.is_isosceles:
            return 2 * self.offset_even
            # return 2 * self._ra * np.sin(np.pi / 2 - self._angle) # ???
        else:
            return self.offset_even + self.offset_odd

    @property
    def offset_even(self):
        """Returns offset between two inner chains. 0-1"""
        return self._ra * np.cos(self._angle)

    @property
    def offset_odd(self):
        """Returns indent between inner chains. 1-2"""
        return (
            self.offset_even if self.is_isosceles else self._rb * np.cos(self.angle_odd)
        )

    @property
    def width(self):
        """Returns distance between two inner chains."""
        return self._ra * np.sin(self._angle)

    @property
    def angle_even(self):
        return self._angle

    @property
    def angle_odd(self):
        if self.is_isosceles:
            return -self._angle
        return -np.arcsin(self.width / self._rb)

    def position(self, i, j):
        """Return distance and angle between chain direction and connection line."""

        if abs(i - j) == 1:
            if i % 2 == 0:
                if i < j:
                    return self._ra, self._angle
                return self.rb, np.pi + self.angle_odd

            if i < j:
                return self.rb, self.angle_odd
            return self._ra, np.pi + self._angle

        if abs(i - j) == 2:
            return self.rc, 0 if i < j else np.pi

        if abs(i - j) % 2 == 0:
            return abs(i - j) // 2 * self.rc, 0 if i < j else np.pi

        if abs(i - j) % 2 == 1:
            ry = self.width
            if i % 2 == 0:
                if i < j:
                    rx = abs(i - j) // 2 * self.rc + self.offset_even
                    angle = np.arctan(ry / rx)
                else:
                    rx = abs(i - j) // 2 * self.rc + self.offset_odd
                    angle = np.pi - np.arctan(ry / rx)
            else:
                if i < j:
                    rx = abs(i - j) // 2 * self.rc + self.offset_odd
                    angle = -np.arctan(ry / rx)
                else:
                    rx = abs(i - j) // 2 * self.rc + self.offset_even
                    angle = np.pi + np.arctan(ry / rx)

            rc = np.sqrt(rx**2 + ry**2)
            return rc, angle

        raise NotImplementedError(f"Unsupported pair ({i}, {j})")
