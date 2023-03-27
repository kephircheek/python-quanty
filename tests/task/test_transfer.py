import json
import unittest
from dataclasses import replace

import numpy as np
import sympy as sp

import quanty.json
import quanty.tools.json
from quanty.geometry import Chain, UniformChain, ZigZagChain
from quanty.hamiltonian import XX, XXZ, Hamiltonian
from quanty.model import Homogeneous, Model
from quanty.problem.transfer import TransferZQCAlongChain
from quanty.task.transfer import ZeroCoherenceTransfer
from quanty.task.transfer_ import FitTransmissionTimeTask, TransferZQCPerfectlyTask

from . import PATH_ASSETS


class TestZeroCoherenceTransferCommon(unittest.TestCase):
    def test_init_classic(self):
        self.h = Hamiltonian(Model(Chain()))
        task = ZeroCoherenceTransfer.init_classic(self.h, 15, n_sender=3, n_ancillas=3)
        self.assertEqual({0, 1, 2}, task.sender)
        self.assertEqual({12, 13, 14}, task.receiver)
        self.assertEqual({9, 10, 11}, task.ancillas)


class TestZeroCoherenceTransferEx1N15S3A3(unittest.TestCase):
    def setUp(self):
        geometry = UniformChain()
        model = Homogeneous(geometry)
        ham = XX(model)
        self.task = ZeroCoherenceTransfer.init_classic(
            ham, length=15, n_sender=3, n_ancillas=3, excitations=1
        )
        self.problem = TransferZQCAlongChain.init_classic(
            ham, length=15, n_sender=3, n_ancillas=3, excitations=1
        )

    def test_fit_transmition_time_task(self):
        task = FitTransmissionTimeTask(self.problem)
        result = task.run()
        self.assertEqual(17.27988, result.transmission_time)

    def test_fit_transmition_time(self):
        self.assertIsNone(self.task.transmission_time)
        self.task.fit_transmission_time(decimals=5)
        self.assertEqual(17.27988, self.task.transmission_time)

    def test_sender_params_number_in_transfer_perfectly_task(self):
        task = TransferZQCPerfectlyTask(self.problem, transmission_time=None)
        self.assertEqual(
            len(task.problem.sender_params), (len(self.task.receiver) - 1) ** 2 + 2
        )

    def test_sender_params_number(self):
        self.assertEqual(
            len(self.task.sender_params), (len(self.task.receiver) - 1) ** 2 + 2
        )

    def test_n_features(self):
        self.assertEqual(
            self.task.n_features,
            len(self.task.ext_receiver) ** 2 - len(self.task.ext_receiver),
        )

    def test_n_equations(self):
        self.assertEqual(
            self.task.n_equations,
            (len(self.task.receiver) - 1) * len(self.task.sender_params) * 2,
        )

    def test_n_residuals(self):
        self.task = replace(self.task, transmission_time=17.27988)
        self.assertEqual(
            self.task.n_equations, len(self.task.perfect_transferred_state_residuals())
        )

    def test_transfer_perfectly_task_reciever_impacts(self):
        with open(PATH_ASSETS / "impacts_n15_s3_a3_ex1.json") as f:
            desired_impacts = json.load(f, object_hook=quanty.tools.json.json2complex)
        task = TransferZQCPerfectlyTask(self.problem, transmission_time=17.27988)
        for p, impact in task.receiver_state_impacts().items():
            with self.subTest(p=p):
                desired_impact = desired_impacts[str(p)]
                np.testing.assert_array_almost_equal(desired_impact, impact, decimal=14)

    def test_reciever_impacts(self):
        self.task = replace(self.task, transmission_time=17.27988)
        with open(PATH_ASSETS / "impacts_n15_s3_a3_ex1.json") as f:
            desired_impacts = json.load(f, object_hook=quanty.tools.json.json2complex)
        for p, impact in self.task.receiver_state_impacts().items():
            with self.subTest(p=p):
                desired_impact = desired_impacts[str(p)]
                np.testing.assert_array_almost_equal(desired_impact, impact, decimal=14)

    def test_perfectly_transfer_system(self):
        with open(PATH_ASSETS / "system_n15_s3_a3_ex1.json") as f:
            desired_system = json.load(f)

        task = TransferZQCPerfectlyTask(self.problem, transmission_time=17.27988)
        system, rpart = task.perfect_transferred_state_system()
        np.testing.assert_array_almost_equal(desired_system, system, decimal=14)

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
        self.task.set_features(data["features"])
        params = list(self.task.perfect_transferred_state_params().values())
        params_desired = data["params"]
        np.testing.assert_array_almost_equal(params_desired, params, decimal=14)

    def test_number_of_extra_prams(self):
        self.assertEqual(
            self.task.n_equations,
            (len(self.task.receiver) - 1) * len(self.task.sender_params) * 2,
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
        self.task = replace(self.task, transmission_time=10.76051)
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


class TestTransferZQCPerfectlyTransferEx3N9S3A0(unittest.TestCase):
    def setUp(self):
        length = 9
        excitations = 3
        n_sender = 3
        tt = 109.7
        geometry = ZigZagChain.from_two_chain(2, 0.3)
        model = Homogeneous(geometry, h_angle=0.74)
        hamiltonian = XXZ(model)
        problem = TransferZQCAlongChain.init_classic(
            hamiltonian=hamiltonian,
            length=length,
            n_sender=n_sender,
            excitations=excitations,
        )
        self.task = TransferZQCPerfectlyTask(problem, transmission_time=tt)

    def test_assert_state_params(self):
        r = self.task.run()
        r.assert_state_params(decimal=14)


class TestDump2JSON(unittest.TestCase):
    def test_dumping(self):
        geometry = UniformChain()
        model = Homogeneous(geometry)
        ham = XX(model)
        problem = TransferZQCAlongChain.init_classic(
            ham, length=9, n_sender=3, excitations=3
        )
        problem_json = quanty.json.dumps(problem)
        problem_loaded = quanty.json.loads(problem_json)
        self.assertEqual(problem_loaded, problem)
