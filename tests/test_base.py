import unittest

import numpy as np
import qutip as qp

from quanty import base


class TestRaisingOperator(unittest.TestCase):
    def test_single(self):
        np.testing.assert_array_almost_equal(base.sup(1).toarray(), qp.sigmap())

    def test_second(self):
        np.testing.assert_array_almost_equal(
            base.sup(2, 1).toarray(), qp.tensor(qp.qeye(2), qp.sigmap())
        )


class TestProjectionOperator:
    def test_commutation_sz_with_total_square_moment(self):
        n = 3
        square = lambda x: x @ x
        ss = square(base.sx(n)) - square(base.sy(n)) + square(base.sz(n))
        sz = base.sz(n)
        commutator = base.commutator(ss, sz)
        np.testing.assert_array_almost_equal(
            commutator.toarray(), base.zeros(n).toarray()
        )
