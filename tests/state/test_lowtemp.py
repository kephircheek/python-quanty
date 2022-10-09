import unittest

import numpy as np

from quanty.basis import ComputationBasis
from quanty.state.lowtemp import init_low_temp_chain

from .. import _quanty


class TestInitLowTemp(unittest.TestCase):
    def test_s3_ex2(self):
        n, ex = 10, 3
        basis = ComputationBasis(n, ex)
        state = np.eye(2) / 2
        rho_actual = init_low_temp_chain(state, basis)
        rho_desired = _quanty.init_low_temp_chain(state, n, ex=ex)
        np.testing.assert_array_equal(rho_actual, rho_desired)
