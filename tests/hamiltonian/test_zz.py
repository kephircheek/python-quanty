import itertools
import unittest

import numpy as np
import qutip as qp

from quanty.basis import ComputationBasis
from quanty.hamiltonian.zz import _zz_pure


def _zz_qutip(n, dc):
    ham = 0
    for i, j in itertools.combinations(range(n), 2):
        zz = [qp.qeye(2)] * n
        zz[i], zz[j] = qp.sigmaz(), qp.sigmaz()
        ham += dc(i, j) * qp.tensor(*zz)

    return ham / 4


class TestHamiltonianzz(unittest.TestCase):
    def test_pure_and_qutip(self):
        n = 5
        dc = lambda i, j: 1 / abs(i - j) ** 3
        left = np.array(_zz_pure(n, dc))
        right = np.array(_zz_qutip(n, dc))
        self.assertTrue(np.allclose(left, right, atol=1e-14))

    def test_pure_and_qutip_ex2(self):
        n = 5
        dc = lambda i, j: 1 / abs(i - j) ** 3
        left = np.array(_zz_pure(n, dc, ex=2))
        high_ex_indices = [
            int(i) for i in set(ComputationBasis(n)) - set(ComputationBasis(n, 2))
        ]
        right = (
            np.delete(
                np.delete(np.array(_zz_qutip(n, dc)), high_ex_indices, 0),
                high_ex_indices,
                1,
            ),
        )
        self.assertTrue(np.allclose(left, right, atol=1e-14))
