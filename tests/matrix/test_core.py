import unittest

import numpy as np

from quanty import base, basis, matrix
from quanty.basis import ComputationBasis

from .. import _quanty


class TestNodeCounting(unittest.TestCase):
    def _test_generator(self, n, ex):
        width = len(basis.ComputationBasis(n, ex))
        self.assertEqual(
            n,
            matrix.count_nodes(np.eye(width).shape, ex=ex),
        )

    def test_n5_ex3(self):
        self._test_generator(5, 3)

    def test_n5_ex1(self):
        self._test_generator(5, 1)


class TestReducing(unittest.TestCase):
    def _test_prjection_operator(self, s):
        n, i = 5, 2
        sz = s(n, i)
        sz_0 = np.array(matrix.reduce(sz, {i}) / 2 ** (n - 1))
        np.testing.assert_equal(sz_0, s().toarray())

    def test_sz(self):
        self._test_prjection_operator(base.sz)

    def test_sx(self):
        self._test_prjection_operator(base.sx)


class TestParameterizedUnitaryMatrix(unittest.TestCase):
    def test_ut(self):
        n = 4
        m = 2**n
        n_features = m**2 - m
        features = np.array([np.random.random() for _ in range(n_features)])
        np.testing.assert_array_almost_equal(
            matrix.unitary_transform_parameterized(features),
            _quanty.unitary_transform_parameterized(features),
            decimal=14,
        )


class TestExtractBlocks(unittest.TestCase):
    def test_coh0_n3_ex1(self):
        elements = {(1, 2), (0, 0), (1, 1), (2, 3), (3, 3), (2, 2), (1, 3)}
        blocks = matrix.get_diagonal_blocks_edges(elements)
        self.assertEqual(2, len(blocks))
        b1, b2 = blocks
        self.assertTupleEqual((slice(0, 1), slice(0, 1)), b1)
        self.assertTupleEqual((slice(1, None), slice(1, None)), b2)

    def test_coh0_n2_ex2(self):
        elements = {(1, 2), (0, 0), (1, 1), (3, 3), (2, 2)}
        blocks = matrix.get_diagonal_blocks_edges(elements)
        self.assertEqual(3, len(blocks))
        b1, b2, b3 = blocks
        self.assertTupleEqual((slice(0, 1), slice(0, 1)), b1)
        self.assertTupleEqual((slice(1, 3), slice(1, 3)), b2)
        self.assertTupleEqual((slice(3, None), slice(3, None)), b3)
