import functools
import math
from dataclasses import dataclass

import numpy as np

from quanty import matrix
from quanty.basis import BaseVector, ComputationBasis
from quanty.geometry import Chain
from quanty.hamiltonian import Hamiltonian
from quanty.state.coherence import coherence_matrix
from quanty.state.lowtemp import init_low_temp_chain


def _positify_indecies(index: int, length: int) -> int:
    """
    >>> positify_indecies(-3, 5)
    2
    """
    if index >= 0 and index < length:
        return index

    if index < 0 and abs(index) <= length:
        return length + index

    raise ValueError("Invalid index")


@dataclass(frozen=True)
class TransferAlongChain:
    """

    Parameters
    ----------
    features:
        parameters of unitary transform on extended receiver
    """

    hamiltonian: Hamiltonian
    length: int
    sender: frozenset[int]
    receiver: frozenset[int]
    ancillas: frozenset[int]
    excitations: int = None

    def __post_init__(self):
        if len(self.sender) != len(self.receiver):
            raise ValueError("sender and receiver should be the same length")

        def normilize_indecies(indecies):
            return frozenset(map(lambda i: _positify_indecies(i, self.length), indecies))

        object.__setattr__(self, "sender", normilize_indecies(self.sender))
        object.__setattr__(self, "receiver", normilize_indecies(self.receiver))
        object.__setattr__(self, "ancillas", normilize_indecies(self.ancillas))

        if (m := max(self.sender | self.receiver | self.ancillas)) >= self.length:
            raise ValueError(f"acquired qubit out of register: {self.m} >= {self.length}")

        # Can be replaced with type annotation?
        if (g := self.hamiltonian.model.geometry) and not isinstance(g, Chain):
            raise ValueError(
                f"state transfer is possible only along a chain, not a {g!r}"
            )

    @classmethod
    def init_classic(
        cls, hamiltonian, length: int, n_sender: int, n_ancillas: int = 0, **kwargs
    ):
        sender_nodes = set(range(n_sender))  # first `k` nodes in chain
        receiver_nodes = set(range(-1, -(n_sender + 1), -1))  # last `k` nodes in chain
        ancillas_nodes = set(range(-(n_sender + 1), -(n_sender + n_ancillas + 1), -1))
        return cls(
            hamiltonian,
            length=length,
            sender=sender_nodes,
            receiver=receiver_nodes,
            ancillas=ancillas_nodes,
            **kwargs,
        )

    @property
    def ex(self):
        return self.excitations

    @functools.cached_property
    def ext_receiver(self):
        return self.receiver | self.ancillas

    @functools.cached_property
    def n_features(self):
        """
        Return number of features of extended receiver unitary transform.
        """
        return sum(
            (lambda x: x**2 - x)(math.comb(len(self.ext_receiver), i + 1))
            for i in range(self.ex)
        )

    @functools.cached_property
    def _feature_bounds_default(self):
        return [(0, 2 * np.pi) for _ in range(self.n_features)]

    @functools.cached_property
    def basis(self):
        return ComputationBasis(self.length, self.ex).sorted_by_excitation()

    @functools.cached_property
    def sender_basis(self):
        basis = ComputationBasis(
            n=len(self.sender), excitations=self.ex
        ).sorted_by_excitation()
        return basis

    @functools.cached_property
    def receiver_basis(self):
        basis = ComputationBasis(len(self.receiver), self.ex).sorted_by_excitation()
        return basis

    @functools.cached_property
    def ext_receiver(self):
        return self.receiver | self.ancillas

    @functools.cached_property
    def ext_receiver_basis(self):
        basis = ComputationBasis(len(self.ext_receiver), self.ex).sorted_by_excitation()
        return basis

    def info(self):
        print(
            f"{self.hamiltonian.model.geometry}:",
            ", ".join(
                sorted([str(i) for i in self.sender])
                + ["..."]
                + [
                    f"({i})" if i not in self.receiver else str(i)
                    for i in sorted(list(self.ext_receiver))
                ]
            ),
        )
        print(
            f"Number of parameters of unitary transform on ext receiver: {self.n_features}",
            # f"Number of parameters of sender state: {len(self.sender_params)}",
            # f"Number of equation to drop extra receiver elements: {self.n_equations}",
            sep=";\n",
        )

    def U(self, transmission_time: float) -> np.ndarray:
        u = self.hamiltonian.U(self.length, transmission_time, ex=self.ex)
        uo = ComputationBasis.reorder_(u, self.basis)
        return uo


@dataclass(frozen=True)
class TransferZQCAlongChain(TransferAlongChain):
    @staticmethod
    def free_corners(mat: matrix.Matrix) -> matrix.Matrix:
        """Heuristic rule to transform non compability matrix."""
        for i in range(mat.shape[0] - 1):
            mat[i, -1] = 0
            mat[-1, i] = 0

        return mat

    def _is_extra_element(self, i, j):
        """
        Heuristic rule for extra elements in zero coherence matrix.
        """
        width = len(self.sender_basis)
        return (i != j) and (((i + 1) == width) or ((j + 1) == width))

    @functools.cached_property
    def sender_extra_params(self):
        variables, _ = coherence_matrix(order=0, basis=self.sender_basis)
        return {
            s: (i, j) for s, (i, j) in variables.items() if self._is_extra_element(i, j)
        }

    @functools.cached_property
    def _sender_state_and_variables(self):
        variables, zero_coherence = coherence_matrix(order=0, basis=self.sender_basis)
        state = self.free_corners(zero_coherence)
        vs = {
            s: (i, j)
            for s, (i, j) in variables.items()
            if not self._is_extra_element(i, j)
        }
        return vs, state

    @functools.cached_property
    def sender_state(self):
        _, state = self._sender_state_and_variables
        return state

    @functools.cached_property
    def sender_params(self):
        vs, _ = self._sender_state_and_variables
        return vs

    @functools.cached_property
    def initial_state(self):
        state = init_low_temp_chain(
            self.sender_state,
            self.basis,
            sender_basis=self.sender_basis,
            dtype=np.object_,
        )
        return state

    @functools.cached_property
    def initial_state_dok(self):
        return tuple(matrix.todok(self.initial_state, up_right=False))
