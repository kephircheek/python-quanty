import unittest

import numpy as np

from quanty.geometry import UniformChain, ZigZagChain
from quanty.model import Homogeneous


class TestHomogeniusSpinChain(unittest.TestCase):
    def setUp(self):
        self.geometry = UniformChain()
        self.model = Homogeneous(self.geometry)

    def test_constatnt(
        self,
    ):
        self.assertEqual(self.model.constant(0, 1), 1)
        self.assertEqual(self.model.constant(1, 3), 0.125)


class TestHomogeneousRectangularChainN3(unittest.TestCase):
    @staticmethod
    def _dc3(i, j, chi):
        phi = np.pi / 4
        d = np.sqrt(2)
        if i == 0 and j == 1:
            return (3 * np.cos(phi - chi) ** 2 - 1) / (d) ** 3 / 2
        elif i == 1 and j == 2:
            return (3 * np.cos(phi + chi) ** 2 - 1) / (d) ** 3 / 2
        elif i == 0 and j == 2:
            return (3 * np.cos(chi) ** 2 - 1) / (2) ** 3 / 2
        raise NotImplementedError

    def setUp(self):
        self.chi = 0.3
        self.dc3 = lambda i, j: self._dc3(i, j, self.chi)
        chain = ZigZagChain(np.pi / 4, np.sqrt(2))
        self.model = Homogeneous(chain, h_angle=self.chi)

    def test_norm(self):
        np.testing.assert_almost_equal(self.model.constant_norm, self.dc3(0, 1))

    def test_dc01(self):
        np.testing.assert_almost_equal(self.model.constant(0, 1), 1)

    def test_dc12(self):
        np.testing.assert_almost_equal(
            self.model.constant(1, 2),
            self.dc3(1, 2) / self.dc3(0, 1),
        )

    def test_dc02(self):
        np.testing.assert_almost_equal(
            self.model.constant(0, 2),
            self.dc3(0, 2) / self.dc3(0, 1),
        )


class TestHomogeneousHambergiteLikeChain(unittest.TestCase):
    def test_even_odd_ratio(self):
        """Bochkin et al. Appl. Magn. Res. 51:667–678 (2020)"""
        h_angle = np.pi / 6
        chain = ZigZagChain(h_angle, 1)
        model = Homogeneous(chain, h_angle=h_angle)
        np.testing.assert_almost_equal(
            abs(model.constant(1, 2) / model.constant(0, 2)), 3 * np.sqrt(3) / 5
        )
