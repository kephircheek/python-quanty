import collections
import functools
import math
import warnings
from typing import Set
from dataclasses import dataclass, asdict

import numpy as np
import sympy as sp
from numba import njit
from scipy import optimize

from quanty import matrix, tools
from quanty.basis import BaseVector, ComputationBasis
from quanty.geometry.chain import Chain
from quanty.hamiltonian import Hamiltonian
from quanty.optimize import brute_random
from quanty.state.coherence import coherence_matrix
from quanty.state.lowtemp import init_low_temp_chain
from quanty.problem.transfer import TransferAlongChain


def as_real_imag(value):
    try:
        return value.as_real_imag()
    except AttributeError:
        return value.real, value.imag


def free_corners(mat: matrix.Matrix) -> matrix.Matrix:
    """Heuristic rule to transform non compability matrix."""
    for i in range(mat.shape[0] - 1):
        mat[i, -1] = 0
        mat[-1, i] = 0

    return mat


def swap_corners(rho: matrix.Matrix) -> matrix.Matrix:
    rho_: matrix.Matrix = rho.copy()
    rho_[0, 0], rho_[-1, -1] = rho_[-1, -1], rho_[0, 0]
    return rho_


@dataclass(frozen=True)
class FitTransmissionTimeTask:
    problem: TransferAlongChain
    decimals: int = 5
    tmin: float = None
    tmax: float = None
    log10_dt: int = 1
    states: BaseVector = None

    def run(self):
        tt, data = fit_transmission_time(
            problem=self.problem,
            decimals=self.decimals,
            tmin=self.tmin,
            tmax=self.tmax,
            log10_dt=self.log10_dt,
            states=self.states,
        )
        return FitTransmissionTimeResult(task=self, transmission_time=tt, data=data)


@dataclass(frozen=True)
class FitTransmissionTimeResult:
    task: FitTransmissionTimeTask
    transmission_time: float
    data: list[tuple[float, float]]


@functools.lru_cache(maxsize=None)
def fit_transmission_time(
    problem: TransferAlongChain, decimals=5, tmin=None, tmax=None, log10_dt=1, states=None
):
    states = states or [
        s
        for s in ComputationBasis(
            len(problem.sender), excitations=problem.ex
        )  # problem.sender_basis?
        if s.excitations == problem.ex
    ]
    system_state = BaseVector(0, problem.length - len(problem.sender))
    basis = ComputationBasis(problem.length, excitations=problem.ex)

    tmin = tmin or problem.length // 2
    tmax = tmax or 2 * problem.length

    i_elements = [
        basis.index(system_state.insert(state, problem.sender)) for state in states
    ]

    t_transmition = None
    data = []
    # for dt in 10.0**np.arange(, (-decimals if decimals > 0 else 0) - 1, -1):
    dt_init = min(int(np.log10(abs(tmin - tmax))), log10_dt)
    for dt in 10.0 ** np.arange(dt_init, (-decimals if decimals > 0 else 0) - 1, -1):
        U_dt = problem.hamiltonian.U(problem.length, dt, ex=problem.ex)
        U = (
            problem.hamiltonian.U(problem.length, tmin, ex=problem.ex)
            if tmin > 0
            else np.eye(len(basis))
        )
        max_loss = 0
        for t in np.arange(tmin, tmax + dt, dt):
            if t > tmax:
                continue
            U = U @ U_dt
            loss = np.sum(
                np.diag(
                    matrix.reduce(
                        sum(
                            (matrix.element_impact(i, i, U) for i in i_elements),
                            np.zeros((len(basis), len(basis))),
                        ),  # digonal elememts are positive
                        problem.receiver,
                        basis=basis,
                        hermitian=True,
                    )
                )[1:]
            ).real
            data.append((t, loss))
            if loss > max_loss:
                max_loss = loss
                t_transmition = t

        delta = 2 * dt
        tmin = (t_transmition - delta) if t_transmition > delta else 0
        tmax = t_transmition + delta

    tt = np.round(t_transmition, decimals)
    return tt, data


@dataclass(frozen=True)
class TransferZQCPerfectlyTask:
    problem: TransferAlongChain
    transmission_time: float
    features = None

    def __post_init__(self):
        object.__setattr__(self, "_receiver_state_impacts", None)

    def _is_extra_element(self, i, j):
        """
        Heuristic rule for extra elements in zero coherence matrix.
        """
        width = len(self.problem.sender_basis)
        return (i != j) and (((i + 1) == width) or ((j + 1) == width))

    @functools.cached_property
    def sender_extra_params(self):
        variables, _ = coherence_matrix(order=0, basis=self.problem.sender_basis)
        return {
            s: (i, j) for s, (i, j) in variables.items() if self._is_extra_element(i, j)
        }

    @functools.cached_property
    def _sender_state_and_variables(self):
        variables, zero_coherence = coherence_matrix(order=0, basis=self.problem.sender_basis)
        state = free_corners(zero_coherence)
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
            self.problem.basis,
            sender_basis=self.problem.sender_basis,
            dtype=np.object_,
        )
        return state

    @functools.cached_property
    def initial_state_dok(self):
        return tuple(matrix.todok(self.initial_state, up_right=False))

    @functools.cached_property
    def _free_symbol_impacts_to_ext_receiver(self):
        U = self.problem.U(self.transmission_time)
        oU = ComputationBasis.reorder_(U, self.problem.basis)
        rho_init_dok = self.initial_state_dok
        return {
            s: matrix.reduce(
                self._free_symbol_impact(s, rho_init_dok, oU),
                self.problem.ext_receiver,
                basis=self.problem.basis,
                subsystem_basis=self.problem.ext_receiver_basis,
                hermitian=False,
            )
            for s in self.sender_params
        }

    @staticmethod
    def _free_symbol_impact(symbol, mat_dok, U):
        return sum(
            (
                coeff * matrix.element_impact(i, j, U)
                for (i, j), coeff in __class__._free_symbol_entries(
                    symbol, mat_dok
                )
            ),
            np.zeros_like(U),
        )

    @staticmethod
    def _free_symbol_entries(symbol, mat_dok):
        """Linear entries expected."""
        return [
            [(i, j), complex(expr.collect(symbol).coeff(symbol))]
            for (i, j), expr in tuple(mat_dok)
        ]

    @property
    def unitary_transform(self):
        """
        Unitary transform to drop unnecessary elements in receiver.
        """
        u = np.eye(len(self.problem.ext_receiver_basis), dtype=complex)
        if self.features is None:
            return u

        if self.ex == 1:
            u_block = matrix.unitary_transform_parameterized(self.features)
            u[1:, 1:] = u_block

        elif self.ex == 2:
            ex1_block_width = len(self.problem.ext_receiver)
            ex1_block_n_features = ex1_block_width**2 - ex1_block_width
            b1_features = self.features[:ex1_block_n_features]
            b1_u = matrix.unitary_transform_parameterized(b1_features)
            u[1 : ex1_block_width + 1, 1 : ex1_block_width + 1] = b1_u

            b2_features = self.features[ex1_block_n_features:]
            b2_u = matrix.unitary_transform_parameterized(b2_features)
            u[ex1_block_width + 1 :, ex1_block_width + 1 :] = b2_u

        else:
            raise ValueError(f"unsupported excitation number: {self.ex}")

        return u

    @functools.cached_property
    def ext_receiver_state_impacts(self):
        if self.transmission_time is None:
            raise ValueError("transmission time is not fitted")

        return collections.OrderedDict(
            sorted(
                tuple(self._free_symbol_impacts_to_ext_receiver.items()),
                key=lambda x: str(x[0])[-2:] + str(x[0])[:-2],
            )
        )

    def ext_receiver_state(self, decimals=None) -> sp.MutableDenseMatrix:
        n = len(self.problem.ext_receiver_basis)
        mat = sp.Matrix(sp.ZeroMatrix(n, n))
        for p, impact in self.ext_receiver_state_impacts.items():
            if decimals is not None:
                impact = impact.round(decimals)
            mat += p * sp.Matrix(impact)
        return mat

    def receiver_state_impacts(self, use_cache=True):
        if self._receiver_state_impacts is not None and use_cache:
            return self._receiver_state_impacts
        elements_impacts = self.ext_receiver_state_impacts

        u = self.unitary_transform
        impacts2r = {}
        for param, impact_ in elements_impacts.items():
            impact = u @ impact_ @ u.conjugate().transpose()
            nodes = {i - self.problem.length for i in self.problem.receiver}
            impacts2r[param] = matrix.reduce(
                impact,
                nodes,
                basis=self.problem.ext_receiver_basis,
                subsystem_basis=self.problem.receiver_basis,
            )

        object.__setattr__(self, "_receiver_state_impacts", impacts2r)
        return impacts2r

    def receiver_state(self, decimals=None, use_cache=True) -> sp.MutableDenseMatrix:
        mat = sp.Matrix(sp.ZeroMatrix(*self.sender_state.shape))
        for p, impact in self.receiver_state_impacts(use_cache=use_cache).items():
            if decimals is not None:
                impact = impact.round(decimals)
            mat += p * sp.Matrix(impact)
        return mat

    def _heuristic_receiver_reversing(self, receiver_rho: matrix.Matrix) -> matrix.Matrix:
        receiver_basis_reversed = self.problem.receiver_basis.reversed()
        return swap_corners(
            self.problem.receiver_basis.reorder(receiver_rho, receiver_basis_reversed)
        )

    def receiver_state_impacts_reversed(self, use_cache=True):
        return collections.OrderedDict(
            (p, self._heuristic_receiver_reversing(i))
            for p, i in self.receiver_state_impacts(use_cache=use_cache).items()
        )

    def receiver_state_reversed(self, decimals=None) -> sp.MutableDenseMatrix:
        return sp.Matrix(
            self._heuristic_receiver_reversing(self.receiver_state(decimals))
        )

    def linearize_matrix_by_sender_params(self, mat):
        # warnings.warn("crutch: use heuristic knowledge about imaginary part name ('y')")
        stack = []
        for p, (i, j) in self.sender_params.items():
            real, image = as_real_imag(mat[i, j])
            if str(p)[0] == "y":
                stack.append(image)
            else:
                stack.append(real)
        return stack


    def perfect_transferred_state_system(self, use_cache=True):
        params_impact = self.receiver_state_impacts_reversed(use_cache=use_cache)
        impacts_stack = []
        for parameter in self.sender_params:
            param_impact = params_impact[parameter]
            impacts_stack.append(self.linearize_matrix_by_sender_params(param_impact))

        A = np.vstack(impacts_stack).T
        A -= np.eye(*A.shape)  # subtract right part
        b = np.zeros(A.shape[0])

        norm = [(1 if i == j else 0) for (i, j) in self.sender_params.values()]
        A[0, :] = norm  # replace equation of x00 with trace rule (trace(rho) = 1)
        b[0] = 1  # the sum equals one
        return A, b

    def perfect_transferred_state_params(self, use_cache=True):  # find_perfect_state
        A, b = self.perfect_transferred_state_system(use_cache=use_cache)
        result = np.linalg.solve(A, b)
        return collections.OrderedDict(zip(self.sender_params, result))

    def perfect_transferred_state_residuals(self, use_cache=True) -> np.ndarray:
        """
        Return sum of extra receiver elements witch should be zero.
        """
        extra_elements_indices = set(self.sender_extra_params.values())

        def as_real_imag(value):
            return value.real, value.imag

        def residuals(mat):
            return sum(
                (as_real_imag(mat[i, j]) for i, j in extra_elements_indices), tuple()
            )

        return np.array(
            sum(
                (
                    residuals(impact)
                    for impact in self.receiver_state_impacts_reversed(
                        use_cache=use_cache
                    ).values()
                ),
                tuple(),
            )
        )

    def run(self):
        residuals = self.perfect_transferred_state_residuals()
        residual_max = None
        if len(residuals) > 0:
            residual_max = max(residuals)
        params = list(self.perfect_transferred_state_params().values())
        return TransferZQCPerfectlyResult(task=self, state_params=params, residual_max=residual_max)


@dataclass(frozen=True)
class TransferZQCPerfectlyResult:
    task: TransferZQCPerfectlyTask
    state_params: list[float]
    residual_max: float = None

