import unittest

from quanty.basis import BaseVector, ComputationBasis

from . import _quanty


class TestComputationBasis(unittest.TestCase):
    def test_basis_fullness(self):
        n = 3
        excitations = 1
        basis = ComputationBasis(n, excitations=excitations)
        self.assertTrue(BaseVector.from_str("010") in basis)
        self.assertEqual(4, len(basis))
        self.assertListEqual([0, 1, 2, 4], list(map(int, basis)))

    def test_sorting_by_excitation(self):
        n = 3
        excitations = 2
        basis = ComputationBasis(n, excitations=excitations)
        basis.sort(key=lambda v: v.excitations)
        self.assertListEqual(list(map(int, basis)), [0, 1, 2, 4, 3, 5, 6])

    def test_reverse_n3_ex1(self):
        n, ex = 3, 1
        reversed_basis = ComputationBasis(n, excitations=ex)
        reversed_basis.reverse()
        self.assertListEqual(
            ["000", "100", "010", "001"],
            list(map(str, reversed_basis)),
        )

    def test_reverse(self):
        n, ex = 3, 1
        basis = ComputationBasis(n, ex)
        basis.reverse()
        basis_desired = [
            _quanty.base_vector(v, n)[::-1] for v in _quanty.computation_basis(n, ex)
        ]
        self.assertListEqual(basis_desired, list(map(str, basis)))


class TestBaseVector(unittest.TestCase):
    def test_hashing(self):
        n = 3
        self.assertEqual(
            hash(BaseVector.from_str("010")), hash(list(ComputationBasis(n, 2))[2])
        )

    def test_insert(self):
        s = BaseVector(31, 5).insert(BaseVector.from_str("11"), {3, 5})
        self.assertEqual(len(s), 7)
        self.assertEqual(s, BaseVector.from_str("1111111"))

    def test_ground_state(self):
        n = 5
        s = BaseVector(0, n)
        self.assertEqual(str(s), "0" * n)
