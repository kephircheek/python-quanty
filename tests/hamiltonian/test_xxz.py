import unittest

import numpy as np

from quanty.basis import ComputationBasis
from quanty.geometry.chain import ZigZagChain
from quanty.hamiltonian.xxz import XXZ
from quanty.model.homo import Homogeneous
from quanty.tools import wolfram_mathematica as wm


class TestHomogeneusZigZagChainWithRightAngle(unittest.TestCase):
    def setUp(self):
        chain = ZigZagChain(np.pi / 4, np.sqrt(2))
        model = Homogeneous(chain, h_angle=0.2)
        self.ham = XXZ(model)

    def test_matrix_n3_ex2(self):
        n, ex = 3, 2
        actual = self.ham(n, ex)
        obasis = ComputationBasis(n, excitations=ex)
        obasis.sort(key=lambda v: v.excitations)
        actual = ComputationBasis.reorder_(actual, obasis)
        data = """{
            {-0.768011, 0, 0, 0, 0, 0, 0},
            {0, -0.231989, -0.0387996, 0.30681, 0, 0, 0},
            {0, -0.0387996, 0.15439, 1/2, 0, 0, 0},
            {0, 0.30681, 1/2, 0.84561, 0, 0, 0},
            {0, 0, 0, 0, 0.84561, 1/2, 0.30681},
            {0, 0, 0, 0, 1/2, 0.15439, -0.0387996},
            {0, 0, 0, 0, 0.30681, -0.0387996, -0.231989}
        }"""
        desire = wm.from_mathematica(data)
        np.testing.assert_array_almost_equal(actual, desire, decimal=6)
