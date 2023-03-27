import math
from dataclasses import dataclass
from typing import List

import numpy as np

from .abc import Geometry


class Chain(Geometry):
    def position(self, i, j):
        """Return distance and angle between chain direction and connection line."""
        raise NotImplementedError


@dataclass(frozen=True)
class UniformChain(Chain):
    r: float = 1

    def position(self, i, j):
        return abs(i - j) * self.r, 0


@dataclass(frozen=True)
class ZigZagChain(Chain):
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

    angle: float
    ra: float
    rb: float = None

    @classmethod
    def from_two_chain(cls, r, width, offset=None):
        """
        Parameters
        ----------
        r: distance between nodes in each chain
        widht: distance between chains
        """
        if offset == None:
            offset = r / 2
            angle = np.arctan(width / offset)
            ra = np.sqrt(width**2 + offset**2)
            return cls(angle=angle, ra=ra)
        raise NotImplementedError

    @property
    def angle_(self):
        max_angle = np.pi / 2
        if abs(angle) >= max_angle:
            warnings.warn(f"angle should be in semi-interval [0, pi/2), not {self.angle}")
            return math.copysign(abs(angle) % max_angle, angle)
        return self.angle

    @property
    def rb_(self):
        return self.ra if self.is_isosceles else self.rb

    @property
    def is_isosceles(self):
        return self.rb is None

    @property
    def rc(self):
        """Returns distance between nearest same parity chain nodes."""
        if self.is_isosceles:
            return 2 * self.offset_even
            # return 2 * self.ra * np.sin(np.pi / 2 - self.angle) # ???
        else:
            return self.offset_even + self.offset_odd

    @property
    def offset_even(self):
        """Returns offset between two inner chains. 0-1"""
        return self.ra * np.cos(self.angle)

    @property
    def offset_odd(self):
        """Returns indent between inner chains. 1-2"""
        return (
            self.offset_even if self.is_isosceles else self.rb_ * np.cos(self.angle_odd)
        )

    @property
    def width(self):
        """Returns distance between two inner chains."""
        return self.ra * np.sin(self.angle)

    @property
    def angle_even(self):
        return self.angle

    @property
    def angle_odd(self):
        if self.is_isosceles:
            return -self.angle
        return -np.arcsin(self.width / self.rb_)

    def position(self, i, j):
        """Return distance and angle between chain direction and connection line."""

        if abs(i - j) == 1:
            if i % 2 == 0:
                if i < j:
                    return self.ra, self.angle
                return self.rb_, np.pi + self.angle_odd

            if i < j:
                return self.rb_, self.angle_odd
            return self.ra, np.pi + self.angle

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
