"""
Removed API.

It was developed for QIP-2022 article and refactored to Quanty package.
"""
import functools
import itertools
import math
import sys
import time as tm
import warnings

import numba as nb
import numpy as np
import sympy as sp
from scipy.special import comb as scipy_combinations


def to_mathematica(matrix: list):
    return str(matrix).replace("[", "{").replace("]", "}").replace("j", "*I")


def from_mathematica(filename):
    with open(filename) as f:
        return np.array(
            eval(
                f.read()
                .replace("{", "[")
                .replace("}", "]")
                .replace("I", "1j")
                .replace("*^", "e")
            )
        )


@functools.cache
def getpackage(cls):
    if issubclass(cls, sp.matrices.common.MatrixRequired):
        return sp

    if issubclass(cls, np.ndarray):
        return np

    raise ValueError(f"unknown class: {cls.__name__}")


@functools.cache
def count_nodes(shape, ex=None):
    if shape[0] != shape[1]:
        raise ValueError(f"expected square matrix, got: {shape}")

    n = np.round(np.log2(shape[0]), 1)

    if ex is not None:
        for i in range(int(n), shape[0]):
            if shape[0] == len(computation_basis(i, ex=ex)):
                return i

    elif abs(n % int(n)) < 1e-15:
        return int(n)

    raise ValueError("could not calculate number of spins")


@functools.cache
def i_base_vector(vector):
    """
    Return number of `vector` in computation basis.

    Examples:
        >>> i_base_vector('010')
        2
    """
    return int(vector, 2)


@functools.cache
def base_vector(i, n):
    """
    Return `i`-th base vecotor  in full basis set in system with `n` particles.

    NB! Numeration statrs from one.

    Examples
    --------
    >>> base_vector(2, 3)
    '001'
    """
    if i < 0:
        raise ValueError("number should be positive")

    return bin(i)[2:].zfill(n)


@functools.cache
def count_exitation(state):
    if isinstance(state, int):
        return count_exitation(bin(state)[2:])
    return sum(map(int, state))


@functools.cache
def combination(n, ex):
    position_combinations = itertools.combinations(range(n), ex)
    return sorted(sum(2**pow for pow in pos) for pos in position_combinations)


@functools.cache
def computation_basis(n, ex=None, only=False):
    """
    Parameters
    ----------
    n (int):
    ex (int): max exitation order
    only (bool): select only setted exitation

    Returns
    -------
        list: of base vectors.

    Examples
    --------
    >>> computation_basis(2)
    [0, 1, 2, 3]
    >>> computation_basis(4, 1)
    [0, 1, 2, 4, 8]
    """
    if n <= 0:
        raise ValueError("system size should be positive integer")

    if ex is None:
        return tuple((range(2**n)))

    if isinstance(ex, int) and ex >= 0:
        if ex > n:
            warnings.warn(f"Exitation higher than max: {ex} > {n}")
        if only:
            return tuple(sorted(combination(n, ex)))

        return tuple(sorted(sum((combination(n, ex) for ex in range(ex + 1)), [])))

    raise ValueError("exitation order should be positive integer or zero")


def insert(system_state, subsystem_state, positions):
    full_state = system_state
    for pos, state in zip(positions, subsystem_state):
        full_state = full_state[:pos] + str(state) + full_state[pos:]
    return full_state


@functools.cache
def reindex(ivector, basis=None):
    if basis is None:
        return ivector
    return list(basis).index(ivector)


def dtype_of(matrix):
    """Return type of data in matrix."""
    if isinstance(matrix, np.ndarray):
        return matrix.dtype.type
    return sp.sympify


def zeros(n, dtype=np.complex128):
    if dtype == sp.sympify:
        return sp.zeros(n)
    return np.zeros((n, n), dtype=dtype)


def zeros_like(mat):
    size = mat.shape[0]
    return zeros(size, dtype=dtype_of(mat))


def _todok(matrix, up_right=False, zeros=False):
    for irow, row in enumerate(matrix):
        for icol, element in enumerate(row):
            if up_right and irow > icol:
                continue

            if not zeros and abs(element) == 0:
                continue

            yield (irow, icol), element


_todok_njit = nb.njit(_todok)


def todok(matrix, up_right=False, zeros=False):
    if isinstance(matrix, np.ndarray) and matrix.dtype.type is not np.object_:
        return _todok_njit(matrix, up_right=up_right, zeros=zeros)

    return _todok(matrix.tolist(), up_right=up_right, zeros=zeros)


def crop(matrix, ex):
    n = count_nodes(matrix.shape)
    basis = computation_basis(n, ex=ex)
    matrix_croped = zeros(len(basis), dtype=dtype_of(matrix))
    for irow, icol in itertools.combinations_with_replacement(basis, 2):
        irow_, icol_ = reindex(irow, basis), reindex(icol, basis)
        matrix_croped[irow_, icol_] = matrix[irow, icol]
        if irow != icol:
            matrix_croped[icol_, irow_] = matrix[icol, irow]

    return matrix_croped


def reorder(rho, basis, ex=None, old_basis=None):
    old_basis = old_basis or computation_basis(count_nodes(rho.shape, ex=ex), ex=ex)
    rho_ = zeros_like(rho)
    for (r, i), (c, j) in itertools.product(enumerate(old_basis), enumerate(old_basis)):
        r_, c_ = basis.index(i), basis.index(j)
        rho_[r_, c_] = rho[r, c]

    return rho_


def groupby(rho, key=None):
    """Grouping elemnts with key function on basis vector."""
    key = key or count_exitation
    o_basis = sorted([v for v in computation_basis(n=3)], key=key)
    return reorder(rho, o_basis)


def reverse_order(rho, ex=None):
    """Reverse spin sequence of state."""
    n = count_nodes(rho.shape, ex=ex)
    basis = [i_base_vector(base_vector(v, n)[::-1]) for v in computation_basis(n, ex)]

    rho_ = reorder(rho, basis, ex=ex)
    return rho_


def swap_corners(rho):
    rho_ = rho.copy()
    rho_[0, 0], rho_[-1, -1] = rho_[-1, -1], rho_[0, 0]
    return rho_


def reduce(
    rho,
    subsystem: set,
    n: int = None,
    ex: int = None,
    hermitian=True,
):
    """
    Args:
        rho (sympy.Matrix, np.ndarray): target matrix
        sub_system (set[int]): spins numbers of sub space
        n (int): number of spins in system
        basis (list[int]):

    NB! numeration starts from zero
    """

    dtype = dtype_of(rho)
    if ex is not None and n is None:
        raise ValueError("if excitation is setted, number of particles is neccessary")

    n = n or count_nodes(rho.shape, ex=ex)
    basis = computation_basis(n, ex=ex)

    if len(basis) != rho.shape[0]:
        raise ValueError("wrong shape of target density matrix")

    subsystem = {i if i >= 0 else (n + i) for i in subsystem}
    if any(ispin >= n or ispin < 0 for ispin in subsystem):
        raise ValueError("spin number should be less than system size")

    m = len(subsystem)  # number of spins in subsystem
    if n == m:
        return rho

    subsystem = sorted(list(subsystem))
    basis_sub = computation_basis(m, ex=ex)
    rho_sub = zeros(len(basis_sub), dtype=dtype)

    def insert_and_reindex(ivec, ivec_sub):
        return reindex(
            i_base_vector(
                insert(base_vector(ivec, n - m), base_vector(ivec_sub, m), subsystem),
            ),
            basis,
        )

    for irow_sub, icol_sub in itertools.product(
        range(len(basis_sub)), range(len(basis_sub))
    ):
        i_state_row_sub = basis_sub[irow_sub]
        i_state_col_sub = basis_sub[icol_sub]
        if hermitian and irow_sub > icol_sub:
            rho_sub[irow_sub, icol_sub] = rho_sub[icol_sub, irow_sub].conjugate()
            continue

        if ex is not None:
            ex_ = ex - max(
                count_exitation(i_state_row_sub), count_exitation(i_state_col_sub)
            )

        else:
            ex_ = None

        basis_remain = computation_basis(n - m, ex=ex_)
        irows = [insert_and_reindex(ivec, i_state_row_sub) for ivec in basis_remain]
        icols = [insert_and_reindex(ivec, i_state_col_sub) for ivec in basis_remain]

        elem = sum(rho[irow, icol] for irow, icol in zip(irows, icols))

        rho_sub[irow_sub, icol_sub] = elem

    return rho_sub


def hxx(n, dc=None, deep=None, ex=None, dtype=np.float64):
    dc = dc or (lambda i, j: 1 / dtype(abs(i - j)) ** 3)
    basis = None if ex is None else computation_basis(n, ex=ex)
    basis_dict = {b: i for i, b in enumerate(basis or [])}
    dim = 2 ** n if basis is None else len(basis)
    ham = zeros(dim, dtype=dtype)

    deep = deep or (n - 1)
    if deep < 1 or deep > (n - 1):
        raise ValueError(
            "dipolar constant deeping must be positive" "and less than particle number"
        )

    def insert_and_reindex(vec, vec_sub, positions):
        vec_ = i_base_vector(insert(vec, vec_sub, positions))
        return basis_dict.get(vec_, vec_)

    for i, j in itertools.combinations_with_replacement(range(n), 2):
        if i == j or abs(i - j) > deep:
            continue

        dc_ = dc(i, j)
        # only |...0...1...><...1...0...| non zero element of I^+I^-
        ex_ = (ex - 1) if ex else None
        basis_ = [0] if ex_ == 0 else computation_basis(n - 2, ex=ex_)
        # irows = [insert_and_reindex(base_vector(ivec, n - 2), '01', [i, j]) for ivec in basis_]
        # icols = [insert_and_reindex(base_vector(ivec, n - 2), '10', [i, j]) for ivec in basis_]
        indeces = (
            (
                insert_and_reindex(base_vector(ivec, n - 2), "01", [i, j]),
                insert_and_reindex(base_vector(ivec, n - 2), "10", [i, j]),
            )
            for ivec in basis_
        )
        for irow, icol in indeces:
            ham[irow, icol] = dc_ / 2
            # if irow != icol:
            ham[icol, irow] = dc_ / 2

    return ham


def _hxx(n, dc=None, deep=None, ex=None, dtype=np.float64):
    dc = dc or (lambda i, j: 1 / dtype(abs(i - j)) ** 3)

    basis = None if ex is None else computation_basis(n, ex=ex)
    dim = 2 ** n if basis is None else len(basis)
    ham = zeros(dim, dtype=dtype)

    deep = deep or (n - 1)
    if deep < 1 or deep > (n - 1):
        raise ValueError(
            "dipolar constant deeping must be positive" "and less than particle number"
        )

    def insert_and_reindex(ivec, vec_sub, positions):
        return reindex(
            i_base_vector(
                insert(base_vector(ivec, n - 2), vec_sub, positions),
            ),
            basis,
        )

    for i, j in itertools.combinations_with_replacement(range(n), 2):
        if i == j or abs(i - j) > deep:
            continue

        # only |...0...1...><...1...0...| non zero element of I^+I^-
        ex_ = (ex - 1) if ex else None
        basis_ = [0] if ex_ == 0 else computation_basis(n - 2, ex=ex_)
        irows = [insert_and_reindex(ivec, "01", [i, j]) for ivec in basis_]
        icols = [insert_and_reindex(ivec, "10", [i, j]) for ivec in basis_]
        for irow, icol in zip(irows, icols):
            dc_ = dc(i, j)
            ham[irow, icol] = dc_ / 2
            if irow != icol:
                ham[icol, irow] = dc_ / 2

    return ham


def coherence_matrix(order, n, ex=None, var=("x", "y"), dtype=sp.sympify):
    basis = computation_basis(n, ex)
    mat = zeros(len(basis), dtype=dtype)
    varibles = []
    for i, j in itertools.combinations_with_replacement(range(len(basis)), 2):
        order_ = abs(count_exitation(basis[i]) - count_exitation(basis[j]))
        if order_ != order:
            continue
        x = sp.Symbol(f"{var[0]}{i}{j}", real=True)
        y = sp.Symbol(f"{var[1]}{i}{j}", real=True)
        varibles.extend([x, y])
        if i == j:
            mat[i, j] = x
        else:
            mat[i, j] = x + sp.I * y
            mat[j, i] = x - sp.I * y

    return varibles, mat


def init_low_temp_chain(rho_sender, n, ex=None, dtype=np.complex128):
    n_sender = count_nodes(rho_sender.shape, ex=ex)
    sender_basis = computation_basis(n_sender, ex=ex)
    basis = computation_basis(n, ex=ex)
    rho_init = zeros(len(basis), dtype=dtype)
    for (i, i_state), (j, j_state) in itertools.product(
        enumerate(sender_basis), enumerate(sender_basis)
    ):
        state = "0" * (n - n_sender)
        irow = reindex(i_base_vector(base_vector(i_state, n) + state), basis)
        icol = reindex(i_base_vector(base_vector(j_state, n) + state), basis)
        rho_init[irow, icol] = rho_sender[i, j]
        if irow != icol:
            rho_init[icol, irow] = rho_sender[j, i]

    return rho_init


def eigh(matrix):
    if isinstance(matrix, np.ndarray):
        return np.linalg.eigh(matrix)

    # can use sp.Matrix.diagonalize() ?
    eigenvals, eigenvecs = [], []
    for eigenval, gmult, subeigvecs in matrix.eigenvects():
        eigenvals.extend([eigenval] * gmult)
        eigenvecs.extend([vec.normalized() for vec in subeigvecs])

    return sp.Matrix(eigenvals), sp.Matrix.hstack(*eigenvecs)


def diag(vector):
    if isinstance(vector, np.ndarray):
        return np.diag(vector)
    return sp.diag(*vector)


def htranspose(matrix):
    return matrix.conjugate().transpose()


def exp(vector):
    if isinstance(vector, np.ndarray):
        return np.exp(vector)
    return sp.Matrix([sp.exp(v) for v in vector])


def qevolution_span(rho, hamiltonian, dt=0, tmin=0, tmax=None):
    [D, V] = eigh(hamiltonian)

    if tmin != 0:
        U = (V @ np.diag(np.exp(-1j * D * tmin)) @ htranspose(V)).round(10)
        rho = U @ rho @ htranspose(U)

    time = tmin
    yield time, rho

    U = V @ np.diag(np.exp(-1j * D * dt)) @ htranspose(V)
    while time <= (tmax - dt) if tmax else True:
        time += dt
        rho = U @ rho @ htranspose(U)
        yield time, rho


def qevolution(rho, hamiltonian):
    """Evolution density matrix with `ham` at `dt` span"""
    [D, V] = eigh(hamiltonian)
    time = sp.Symbol("t", real=True, nonnegative=True)
    e_iDt = sp.diag(*sp.Matrix(-sp.I * D * time).applyfunc(sp.exp))
    U = V @ e_iDt @ htranspose(V)
    rho = U @ rho @ htranspose(U)
    return time, rho


def element_impact(i, j, U, k=None, m=None):
    if k is None and m is None:
        return U[:, i].reshape(-1, 1) @ U[:, j].conjugate().reshape(1, -1)

    if m is None:
        return U[k, i] * U[:, j].conjugate().reshape(1, -1)

    if k is None:
        return U[:, i].reshape(-1, 1) * U[m, j].conjugate()

    return U[k, i] * U[m, j].conjugate()


def linsolve(equations, varibles):
    solutions = sp.linsolve(equations, varibles)
    if len(solutions) < 1:
        raise ValueError("no solution")

    solution = next(iter(solutions))
    return {k: v.evalf() for k, v in zip(varibles, solution)}


@nb.njit(nb.f8(nb.u4, nb.u4, nb.u4))
def transmition_line_dc(i, j, n):
    """Dipolar coupling constant between chain nodes with 2 pairs of adjusted coupling."""
    D01, D12 = 0.3005, 0.5311

    if j < i:
        i, j = j, i

    if i < n and j < n and i >= 0 and j >= 0:
        r01 = (1 / D01) ** (1 / 3)
        r12 = (1 / D12) ** (1 / 3)
        distances = np.array([r01, r12] + [1.0] * (n - 5) + [r12, r01])
        return 1 / (np.sum(distances[i:j]) ** 3)

    raise ValueError("bad indexes")


@nb.njit(nb.f8(nb.u4, nb.u4, nb.u4))
def uniform_chain_dc(i, j, n):
    """Dipolar coupling constant between chain nodes with 2 pairs of adjusted coupling."""
    if j < i:
        i, j = j, i

    if i < n and j < n and i >= 0 and j >= 0:
        return 1 / abs(j - i) ** 3

    raise ValueError("bad indexes")


@functools.cache
def calc_transmission_time(
    n,
    hamiltonian=None,
    *,
    sender={0},
    reciever={-1},
    ex=None,
    decimals=5,
    tmin=None,
    tmax=None,
    states=None,
    return_data=False,
):
    states = states or [
        base_vector(s, len(sender))
        for s in computation_basis(len(sender), ex=1, only=True)
    ]
    system_state = "0" * (n - len(sender))
    basis = computation_basis(n, ex)
    tmin = tmin or n // 2
    tmax = tmax or 2 * n

    if hamiltonian is None:
        dc = lambda i, j: uniform_chain_dc(i, j, n)
        hamiltonian = hxx(n, dc=dc, ex=ex, dtype=np.float64)
    eigvals, eigvecs = eigh(hamiltonian)
    eigvals = eigvals.astype(np.complex128)
    ut = lambda t: eigvecs @ np.diag(np.exp(-1j * eigvals * t)) @ eigvecs.conj().T

    i_elements = [
        basis.index(i_base_vector(insert(system_state, state, sender)))
        for state in states
    ]

    t_transmition = None
    data = []
    # for dt in 10.0**np.arange(, (-decimals if decimals > 0 else 0) - 1, -1):
    time_order = int(np.log10(abs(tmin - tmax)))
    dt_init = time_order if time_order < 1 else 1
    for dt in 10.0 ** np.arange(dt_init, (-decimals if decimals > 0 else 0) - 1, -1):
        U_dt = ut(dt)
        U = ut(tmin) if tmin > 0 else np.eye(len(basis))
        max_loss = 0
        for t in np.arange(tmin, tmax + dt, dt):
            if t > tmax:
                continue
            U = U @ U_dt
            loss = np.sum(
                np.diag(
                    reduce(
                        sum(
                            (element_impact(i, i, U) for i in i_elements),
                            np.zeros((len(basis), len(basis))),
                        ),  # digonal elememts are positive
                        reciever,
                        n=n,
                        ex=ex,
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

    if return_data:
        return np.round(t_transmition, decimals), data
    return np.round(t_transmition, decimals)


@nb.njit
def unitary_transform_parameterized(x):
    def bm(size, i, j, phi):
        mat = np.eye(size, dtype=np.complex128)
        mat[i, i] = mat[j, j] = np.cos(phi)
        if i < j:
            mat[i, j] = -np.sin(phi)
            mat[j, i] = -mat[i, j]
        else:
            mat[i, j] = mat[j, i] = 1j * np.sin(phi)
        return mat

    size_ = (1 + np.sqrt(1 + 4 * len(x))) / 2
    size = int(size_)
    if np.abs(size_ - size) > 1e-14:
        raise ValueError(f"wrong number of parameters")

    result = np.eye(size, dtype=np.complex128)
    i_param = 0
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            result = result @ bm(size, i, j, x[i_param])
            i_param += 1

    return result
