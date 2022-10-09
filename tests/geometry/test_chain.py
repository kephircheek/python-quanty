import unittest

import numpy as np

from quanty.geometry.chain import UniformChain, ZigZagChain


class TestUniformChain(unittest.TestCase):
    def setUp(self):
        self.chain = UniformChain()

    def test_distance(self):
        r, _ = self.chain.position(1, 4)
        self.assertEqual(3, r)


class TestZigZagChainWithRightAngle(unittest.TestCase):
    def setUp(self):
        self.chain = ZigZagChain(np.pi / 4, 1)

    def test_position_02(self):
        r, a = self.chain.position(0, 2)
        np.testing.assert_almost_equal(r, np.sqrt(2))
        np.testing.assert_almost_equal(a, 0)

    def test_position_12(self):
        r, a = self.chain.position(1, 2)
        np.testing.assert_almost_equal(r, self.chain.ra)
        np.testing.assert_almost_equal(a, -np.pi / 4)

    def test_position_01(self):
        r, a = self.chain.position(0, 1)
        np.testing.assert_almost_equal(r, self.chain.ra)
        np.testing.assert_almost_equal(a, np.pi / 4)


class TestZigZagChainWithZeroAngle(unittest.TestCase):
    def setUp(self):
        self.chain = ZigZagChain(0, 1)

    def test_position_second_neigbour(self):
        r, a = self.chain.position(0, 2)
        np.testing.assert_almost_equal(r, 2 * self.chain.ra)
        np.testing.assert_almost_equal(a, 0)


class TestZigZagChainNotIsosceles(unittest.TestCase):
    def setUp(self):
        self.chain = ZigZagChain(np.pi / 4, 3, 4)

    def test_geometry(self):
        def polar2dec(r, theta, shift_x=0):
            return r * np.cos(theta) + shift_x, r * np.sin(theta)

        def nodes_coords(m, n):
            return (self.chain.position(m, i) for i in range(n))

        points_from_1 = [polar2dec(*point) for point in nodes_coords(1, 10)]
        points_from_3_shifted = [
            polar2dec(*point, shift_x=2 * self.chain.rc) for point in nodes_coords(5, 10)
        ]
        np.testing.assert_array_almost_equal(
            points_from_1, points_from_3_shifted, decimal=14
        )

        points_from_0 = [polar2dec(*point) for point in nodes_coords(2, 10)]
        points_from_2_shifted = [
            polar2dec(*point, shift_x=2 * self.chain.rc) for point in nodes_coords(6, 10)
        ]
        np.testing.assert_array_almost_equal(
            points_from_0, points_from_2_shifted, decimal=14
        )
