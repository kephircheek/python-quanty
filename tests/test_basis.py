import json
import math
import unittest

from quanty.basis import BaseVector, ComputationBasis
import quanty.json

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
        basis = ComputationBasis(n, excitations=excitations).sorted_by_excitation()
        self.assertListEqual(list(map(int, basis)), [0, 1, 2, 4, 3, 5, 6])

    def test_reverse_n3_ex1(self):
        n, ex = 3, 1
        reversed_basis = ComputationBasis(n, excitations=ex).reversed()
        self.assertListEqual(
            ["000", "100", "010", "001"],
            list(map(str, reversed_basis)),
        )

    def test_reverse(self):
        n, ex = 3, 1
        basis = ComputationBasis(n, ex).reversed()
        basis_desired = [
            _quanty.base_vector(v, n)[::-1] for v in _quanty.computation_basis(n, ex)
        ]
        self.assertListEqual(basis_desired, list(map(str, basis)))

    def test_large_basis_size(self):
        n, ex = 256, 2
        basis = ComputationBasis(n, ex)
        self.assertEqual(1 + n + math.comb(n, 2), len(basis))

    def test_jsonify_default(self):
        b = ComputationBasis(5, 3)
        actual_dump_str = quanty.json.dumps(b, indent=2)
        actual_dump_obj = json.loads(actual_dump_str)
        desired_dump_obj = {
            "__class__": {"__module__": "quanty.basis", "__name__": "ComputationBasis"},
            "__init__": [[], {"n": 5, "excitations": 3}],
        }
        self.assertDictEqual(desired_dump_obj, actual_dump_obj)

    def test_jsonify_reversed(self):
        self.maxDiff = None
        b = ComputationBasis(2, 1).reversed()
        actual_dump_str = quanty.json.dumps(b, indent=2)
        actual_dump_obj = json.loads(actual_dump_str)
        desired_dump_obj = {
            "__class__": {"__module__": "quanty.basis", "__name__": "ComputationBasis"},
            "__init__": [
                [],
                {
                    "n": 2,
                    "excitations": 1,
                    "vectors": [
                        {
                            "__class__": {
                                "__module__": "quanty.basis",
                                "__name__": "BaseVector",
                            },
                            "__init__": [[], {'n': 2, 'vector': 0}],
                        },
                        {
                            "__class__": {
                                "__module__": "quanty.basis",
                                "__name__": "BaseVector",
                            },
                            "__init__": [[], {'n': 2, 'vector': 2}],
                        },
                        {
                            "__class__": {
                                "__module__": "quanty.basis",
                                "__name__": "BaseVector",
                            },
                            "__init__": [[], {'n': 2, 'vector': 1}],
                        },
                    ],
                },
            ],
        }
        self.assertDictEqual(desired_dump_obj, actual_dump_obj)


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
