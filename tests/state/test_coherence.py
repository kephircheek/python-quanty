import unittest

from quanty import matrix
from quanty.basis import ComputationBasis
from quanty.state.coherence import coherence_matrix

from .. import _quanty


class TestZeroCoherence(unittest.TestCase):
    """
    Since '_quanty' collect uselessness image part of diagonal element,
    that always zero, the number of variables are not the same with it
    """

    def test_n2(self):
        n = 2
        basis = ComputationBasis(n)
        variables, mat = coherence_matrix(0, basis)
        _, mat_ = _quanty.coherence_matrix(0, n)
        self.assertEqual(len(variables), 6)
        self.assertEqual(mat, mat_)

    def test_n3_ex1(self):
        n, ex = 3, 1
        basis = ComputationBasis(n, ex)
        variables, mat = coherence_matrix(0, basis)
        _, mat_ = _quanty.coherence_matrix(0, n, ex)
        self.assertEqual(mat, mat_)

    def test_n3_ex2_ordered_by_ex(self):
        n, ex = 3, 2
        obasis = ComputationBasis(n, ex).sorted_by_excitation()
        variables, mat_ordered = coherence_matrix(0, obasis)
        self.assertEqual(len(matrix.get_diagonal_blocks_edges(variables.values())), 3)


class TestExtractBlocks(unittest.TestCase):
    def _blocks_of_coh0(self, n, ex):
        basis = ComputationBasis(n, ex)
        variables, mat = coherence_matrix(0, basis)
        elements = {(i, j) for i, j in variables.values()}
        return matrix.get_diagonal_blocks_edges(elements)

    def test_coh0_n3_ex1(self):
        blocks = self._blocks_of_coh0(3, 1)
        self.assertEqual(2, len(blocks))
        b1, b2 = blocks
        self.assertTupleEqual((slice(0, 1), slice(0, 1)), b1)
        self.assertTupleEqual((slice(1, None), slice(1, None)), b2)

    def test_coh0_n2_ex2(self):
        blocks = self._blocks_of_coh0(2, 2)
        self.assertEqual(3, len(blocks))
        b1, b2, b3 = blocks
        self.assertTupleEqual((slice(0, 1), slice(0, 1)), b1)
        self.assertTupleEqual((slice(1, 3), slice(1, 3)), b2)
        self.assertTupleEqual((slice(3, None), slice(3, None)), b3)
