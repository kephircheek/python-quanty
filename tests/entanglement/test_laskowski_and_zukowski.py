import unittest

import numpy as np

from quanty.entanglement.laskowski_and_zukowski import *


class TestLaskowskiAndZukowski(unittest.TestCase):
    def test_ghz(self):
        n = 3
        ghz = np.zeros((2**n, 2**n))
        ghz[0, 0] = 1 / 2
        ghz[-1, 0] = 1 / 2
        ghz[0, -1] = 1 / 2
        ghz[-1, -1] = 1 / 2
        self.assertTrue(3, k_entangled(ghz))

    def test_cat_with_ancilla(self):
        cat = np.zeros((2**2, 2**2))
        cat[0, 0] = 1 / 2
        cat[-1, 0] = 1 / 2
        cat[0, -1] = 1 / 2
        cat[-1, -1] = 1 / 2

        ancilla = np.zeros((2, 2))
        ancilla[0, 0] = 1 / 2
        ancilla[-1, 0] = 1 / 2
        ancilla[0, -1] = 1 / 2
        ancilla[-1, -1] = 1 / 2

        rho = np.kron(cat, ancilla)
        self.assertEqual(2, k_entangled(rho))

    def test_small_antidiagonal(self):
        n = 2
        state = np.zeros((2**n, 2**n))
        state[-1, 0] = (1 / 2) ** (n + 1)
        with self.assertRaisesRegex(ValueError, "antidiagonal element too small"):
            k_separable(state)

    def test_zero_antidiagonal(self):
        n = 2
        state = np.zeros((2**n, 2**n))
        with self.assertRaisesRegex(ValueError, "all antidiagonal elements are zero."):
            k_separable(state)
