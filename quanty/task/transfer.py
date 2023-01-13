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
class ZeroCoherenceTransfer:
    """

    Parameters
    ----------
    features:
        parameters of unitary transform on extended receiver
    """

    hamiltonian: Hamiltonian
    length: int
    sender: set[int]
    receiver: set[int]
    ancillas: set[int]
    excitations: int = None
    transmission_time: int = None
    features = None

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

        if (g := self.hamiltonian.model.geometry) and not isinstance(g, Chain):
            raise ValueError(
                f"state transfer is possible only along a chain, not a {g!r}"
            )

        if self.features is not None and (n := len(self.features)) != self.n_features:
            raise ValueError(
                "wrong number of unitary transform parameters: "
                f"{self.n_features} != {n}"
            )
        object.__setattr__(self, "_features_history", [])
        object.__setattr__(self, "_receiver_state_impacts", None)

    @property
    def ex(self):
        return self.excitations

    def set_features(self, values):
        if np.any(self.features != values):
            object.__setattr__(self, "features", values)
            self._features_history.append(self.features)
            object.__setattr__(self, "_receiver_state_impacts", None)

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

    @functools.cached_property
    def basis(self):
        return ComputationBasis(self.length, self.ex).sorted_by_excitation()

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

    @functools.cached_property
    def n_features(self):
        return sum(
            (lambda x: x**2 - x)(math.comb(len(self.ext_receiver), i + 1))
            for i in range(self.ex)
        )

    @functools.cached_property
    def _feature_bounds_default(self):
        return [(0, 2 * np.pi) for _ in range(self.n_features)]

    def load(self, data):
        data["hamiltonian"].pop("__class__")

    def info(self):
        print(
            f"{self.hamiltonian.model.geometry.__class__.__name__}:",
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
            f"Length of chain (n): {self.length}",
            f"Number of parameters of unitary transform on ext receiver: {self.n_features}",
            f"Number of parameters of sender state: {len(self.sender_params)}",
            f"Number of equation to drop extra receiver elements: {self.n_equations}",
            sep=";\n",
        )

    @functools.cache
    def fit_transmission_time(
        self, decimals=5, tmin=None, tmax=None, log10_dt=1, states=None
    ):
        states = states or [
            s
            for s in ComputationBasis(len(self.sender), excitations=self.ex)
            if s.excitations == self.ex
        ]
        system_state = BaseVector(0, self.length - len(self.sender))
        basis = ComputationBasis(self.length, excitations=self.ex)

        tmin = tmin or self.length // 2
        tmax = tmax or 2 * self.length

        i_elements = [
            basis.index(system_state.insert(state, self.sender)) for state in states
        ]

        t_transmition = None
        data = []
        # for dt in 10.0**np.arange(, (-decimals if decimals > 0 else 0) - 1, -1):
        dt_init = min(int(np.log10(abs(tmin - tmax))), log10_dt)
        for dt in 10.0 ** np.arange(dt_init, (-decimals if decimals > 0 else 0) - 1, -1):
            U_dt = self.hamiltonian.U(self.length, dt, ex=self.ex)
            U = (
                self.hamiltonian.U(self.length, tmin, ex=self.ex)
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
                            self.receiver,
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
        object.__setattr__(self, "transmission_time", tt)
        return data

    @property
    def n_equations(self):
        """Returns number of extra elements residuals."""
        return len(self.sender_extra_params.values()) * len(self.sender_params)

    @staticmethod
    def _free_symbol_entries(symbol, mat_dok):
        """Linear entries expected."""
        return [
            [(i, j), complex(expr.collect(symbol).coeff(symbol))]
            for (i, j), expr in tuple(mat_dok)
        ]

    @staticmethod
    def _free_symbol_impact(symbol, mat_dok, U):
        return sum(
            (
                coeff * matrix.element_impact(i, j, U)
                for (i, j), coeff in ZeroCoherenceTransfer._free_symbol_entries(
                    symbol, mat_dok
                )
            ),
            np.zeros_like(U),
        )

    @functools.cached_property
    def _free_symbol_impacts_to_ext_receiver(self):
        U = self.hamiltonian.U(self.length, self.transmission_time, ex=self.ex)
        oU = ComputationBasis.reorder_(U, self.basis)
        rho_init_dok = self.initial_state_dok
        return {
            s: matrix.reduce(
                self._free_symbol_impact(s, rho_init_dok, oU),
                self.ext_receiver,
                basis=self.basis.copy(),
                subsystem_basis=self.ext_receiver_basis.copy(),
                hermitian=False,
            )
            for s in self.sender_params
        }

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

    @property
    def unitary_transform(self):
        """
        Unitary transform to drop unnecessary elements in receiver.
        """
        u = np.eye(len(self.ext_receiver_basis), dtype=complex)
        if self.features is None:
            return u

        if self.ex == 1:
            u_block = matrix.unitary_transform_parameterized(self.features)
            u[1:, 1:] = u_block

        elif self.ex == 2:
            ex1_block_width = len(self.ext_receiver)
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
        n = len(self.ext_receiver_basis)
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
            nodes = {i - self.length for i in self.receiver}
            impacts2r[param] = matrix.reduce(
                impact,
                nodes,
                basis=self.ext_receiver_basis.copy(),
                subsystem_basis=self.receiver_basis.copy(),
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
        receiver_basis_reversed = self.receiver_basis.reversed()
        return swap_corners(
            self.receiver_basis.reorder(receiver_rho, receiver_basis_reversed)
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

    def perfect_transferred_state(self):
        params = self.perfect_transferred_state_params()
        return sp.Matrix(self.sender_state).evalf(subs=params)

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

    def fit_transfer(
        self,
        loss,
        method,
        method_kwargs={},
        polish=True,
        fsolve=True,
        verbose=True,
    ):
        method_kwargs = method_kwargs.copy()

        def loss_residual(features):
            self.set_features(features)
            return np.max(np.abs(self.perfect_transferred_state_residuals()))

        def loss_target(features):
            self.set_features(features)
            return loss(np.array(self.perfect_transferred_state(), dtype=complex))

        def loss_general(features):
            residual_weight = 1
            residual = loss_residual(features)
            loss_ = loss_target(features)
            full_loss = residual_weight * residual + loss_
            # print(f"residual = {residual} loss = {loss_}")
            return full_loss

        def print_status():
            residual = loss_residual(self.features)
            loss_ = loss_target(self.features)
            print(f"residual = {residual:.0e} loss = {loss_:.1f}")

        if verbose:

            def _dummy_callback(*args, **kwargs):
                pass

            callback = method_kwargs.get("callback", _dummy_callback)

            def _callback(*args, **kwargs):
                print_status()
                return callback(*args, **kwargs)

            method_kwargs["callback"] = _callback

        if method == "dual_annealing":
            method_kwargs.setdefault("bounds", self._feature_bounds_default)
            res = optimize.dual_annealing(loss_general, **method_kwargs)
            self.set_features(res.x)

        elif method == "brute_random":
            method_kwargs.setdefault("ranges", self._feature_bounds_default)
            res = brute_random(loss_general, verbose=verbose, **method_kwargs)
            self.set_features(res.x)

        else:
            raise ValueError(f"unsupported method: {method}")

        if polish:
            print("[polish for fsolve]")
            res = optimize.minimize(loss_residual, self.features, method="L-BFGS-B")
            self.set_features(res.x)
            print_status()

        if fsolve:
            print("[fsolve]")

            def residuals(features):
                self.set_features(features)
                r = self.perfect_transferred_state_residuals(use_cache=False)  # why?
                return r

            if self.n_equations > self.n_features:
                self.set_features(
                    optimize.fsolve(
                        lambda x: residuals(x[: self.n_features]),
                        np.hstack(
                            (self.features, [0] * (self.n_equations - self.n_features))
                        ),
                    )[: self.n_features]
                )
            else:
                self.set_features(
                    optimize.fsolve(
                        lambda x: np.hstack(
                            (residuals(x), [0] * (self.n_features - self.n_equations))
                        ),
                        self.features,
                    )
                )
            print_status()

    def test_perfect_tras(self):
        """Return"""
