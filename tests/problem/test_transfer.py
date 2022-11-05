import unittest

from quanty.geometry import Chain
from quanty.model import Homogeneous, Model
from quanty.hamiltonian import Hamiltonian
from quanty.problem.transfer import TransferAlongChain


class TestTransferAlongChain(unittest.TestCase):
    def test_init_classic(self):
        h = Hamiltonian(Model(Chain()))
        problem = TransferAlongChain.init_classic(h, 15, n_sender=3, n_ancillas=3)
        self.assertEqual({0, 1, 2}, problem.sender)
        self.assertEqual({12, 13, 14}, problem.receiver)
        self.assertEqual({9, 10, 11}, problem.ancillas)
