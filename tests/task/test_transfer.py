import json
import unittest

import numpy as np
import sympy as sp

import quanty.tools.json
from quanty.geometry import Chain, UniformChain
from quanty.hamiltonian import XX, Hamiltonian
from quanty.model import Homogeneous, Model
from quanty.task.transfer import ZeroCoherenceTransfer

from . import PATH_ASSETS


class TestZeroCoherenceTransferCommon(unittest.TestCase):
    def test_init_classic(self):
        self.h = Hamiltonian(Model(Chain()))
        task = ZeroCoherenceTransfer.init_classic(self.h, 15, n_sender=3, n_ancillas=3)
        self.assertEqual({0, 1, 2}, task.sender_nodes)
        self.assertEqual({12, 13, 14}, task.receiver_nodes)
        self.assertEqual({9, 10, 11}, task.ancilla_nodes)


class TestZeroCoherenceTransferEx1N15S3A3(unittest.TestCase):
    def setUp(self):
        geometry = UniformChain()
        model = Homogeneous(geometry)
        ham = XX(model)
        self.task = ZeroCoherenceTransfer.init_classic(
            ham, length=15, n_sender=3, n_ancillas=3, excitations=1
        )

    def test_fit_transmition_time(self):
        self.assertIsNone(self.task.transmission_time)
        self.task.fit_transmission_time(decimals=5)
        self.assertEqual(17.27988, self.task.transmission_time)

    def test_sender_params_number(self):
        self.assertEqual(
            len(self.task.sender_params), (len(self.task.receiver_nodes) - 1) ** 2 + 2
        )

    def test_n_features(self):
        self.assertEqual(
            self.task.n_features,
            len(self.task.extended_receiver_nodes) ** 2
            - len(self.task.extended_receiver_nodes),
        )

    def test_n_equations(self):
        self.assertEqual(
            self.task.n_equations,
            (len(self.task.receiver_nodes) - 1) * len(self.task.sender_params) * 2,
        )

    def test_n_residuals(self):
        self.task._tt = 17.27988
        self.assertEqual(
            self.task.n_equations, len(self.task.perfect_transferred_state_residuals())
        )

    def test_reciever_impacts(self):
        self.task._tt = 17.27988
        with open(PATH_ASSETS / "impacts_n15_s3_a3_ex1.json") as f:
            desired_impacts = json.load(f, object_hook=quanty.tools.json.json2complex)
        for p, impact in self.task.receiver_state_impacts().items():
            with self.subTest(p=p):
                desired_impact = desired_impacts[str(p)]
                np.testing.assert_array_almost_equal(desired_impact, impact, decimal=14)

    def test_system(self):
        self.task.fit_transmission_time()
        with open(PATH_ASSETS / "system_n15_s3_a3_ex1.json") as f:
            desired_system = json.load(f)

        system, rpart = self.task.perfect_transferred_state_system()
        np.testing.assert_array_almost_equal(desired_system, system, decimal=14)

    def test_perfect_state(self):
        self.task.fit_transmission_time()
        with open(PATH_ASSETS / "perfect_transferred_state_ex1_n3_a3.json") as f:
            data = json.load(f)
        self.task.features = data["features"]
        params = list(self.task.perfect_transferred_state_params().values())
        params_desired = data["params"]
        np.testing.assert_array_almost_equal(params_desired, params, decimal=14)

    def test_number_of_extra_prams(self):
        self.assertEqual(
            self.task.n_equations,
            (len(self.task.receiver_nodes) - 1) * len(self.task.sender_params) * 2,
        )


class TestZeroCoherenceTransferEx2N15S3A3(unittest.TestCase):
    def setUp(self):
        geometry = UniformChain()
        model = Homogeneous(geometry)
        ham = XX(model)
        self.task = ZeroCoherenceTransfer.init_classic(
            ham, length=15, n_sender=3, n_ancillas=3, excitations=2
        )

    def test_linearising_by_sender_params(self):
        self.assertListEqual(
            list(self.task.sender_params.keys()),
            self.task.linearize_matrix_by_sender_params(self.task.sender_state),
        )

    def test_n_features(self):
        self.assertEqual(240, self.task.n_features)


class TestZeroCoherenceTransferEx3N9S3A0(unittest.TestCase):
    def setUp(self):
        geometry = UniformChain()
        model = Homogeneous(geometry)
        ham = XX(model)
        self.task = ZeroCoherenceTransfer.init_classic(
            ham, length=9, n_sender=3, excitations=3
        )

    def test_fit_transmition_time(self):
        self.task.fit_transmission_time()
        np.testing.assert_almost_equal(self.task.transmission_time, 10.76051)

    def test_perfect_transfer_without_unitary_transform(self):
        self.task._tt = 10.76051
        np.testing.assert_array_equal(
            np.zeros(self.task.n_equations),
            self.task.perfect_transferred_state_residuals(),
        )
        params = self.task.perfect_transferred_state_params()
        np.testing.assert_array_almost_equal(
            self.task.receiver_state_reversed(14).evalf(3, subs=params),
            sp.Matrix(self.task.sender_state).evalf(3, subs=params),
            decimal=14,
        )
