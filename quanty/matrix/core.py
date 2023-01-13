import enum
import functools
import itertools
from typing import Any, TypeAlias

import numba as nb
import numpy as np
import sympy as sp

from quanty.basis import ComputationBasis

Matrix: TypeAlias = np.ndarray | sp.MutableDenseMatrix


class Type(enum.Flag):
    NUMPY_NDARRAY = "numpy.ndarray"
    SYMPY_MATRIX = "sumpy.Matrix"


###
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


def zeros_like(mat):
    size = mat.shape[0]
    return zeros(size, dtype=dtype_of(mat))


def zeros(n, dtype=np.complex128):
    if dtype == sp.sympify:
        return sp.zeros(n)
    return np.zeros((n, n), dtype=dtype)


def crop(matrix, ex, n=None):
    n = n or count_nodes(matrix.shape)
    basis = computation_basis(n, ex=ex)
    matrix_croped = zeros(len(basis), dtype=dtype_of(matrix))
    for irow, icol in itertools.combinations_with_replacement(basis, 2):
        irow_, icol_ = reindex(irow, basis), reindex(icol, basis)
        matrix_croped[irow_, icol_] = matrix[irow, icol]
        if irow != icol:
            matrix_croped[icol_, irow_] = matrix[icol, irow]

    return matrix_croped


####


def dtype_of(matrix):
    """Return type of data in matrix."""
    if isinstance(matrix, np.ndarray):
        return matrix.dtype.type
    return sp.sympify


def reduce(
    rho,
    subsystem: set,
    basis=None,
    subsystem_basis=None,
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
    # if ex is not None and n is None:
    #     raise ValueError("if excitation is setted, number of particles is neccessary")

    basis = basis or ComputationBasis(count_nodes(rho.shape))

    if len(basis) != rho.shape[0]:
        raise ValueError("wrong shape of target density matrix")

    subsystem = {i if i >= 0 else (basis.n + i) for i in subsystem}
    if any(ispin >= basis.n or ispin < 0 for ispin in subsystem):
        raise ValueError("spin number should be less than system size")

    m = len(subsystem)  # number of spins in subsystem
    if basis.n == m:
        return rho

    subsystem_ordered = sorted(list(subsystem))
    subsystem_basis = subsystem_basis or ComputationBasis(m, excitations=basis.ex)
    rho_sub = zeros(len(subsystem_basis), dtype=dtype)

    for rstate, cstate in itertools.product(subsystem_basis, subsystem_basis):
        irstate, icstate = subsystem_basis.index(rstate), subsystem_basis.index(cstate)
        if hermitian and irstate > icstate:
            rho_sub[irstate, icstate] = rho_sub[icstate, irstate].conjugate()
            continue

        if basis.ex is not None:
            ex_ = basis.ex - max(rstate.excitations, cstate.excitations)

        else:
            ex_ = None

        basis_remain = ComputationBasis(basis.n - m, excitations=ex_)
        common_states = (
            (s.insert(rstate, subsystem_ordered), s.insert(cstate, subsystem_ordered))
            for s in basis_remain
        )

        elem = sum(rho[basis.index(rs), basis.index(cs)] for rs, cs in common_states)

        rho_sub[irstate, icstate] = elem

    return rho_sub


def element_impact(i, j, U, k=None, m=None):
    if k is None and m is None:
        return U[:, i].reshape(-1, 1) @ U[:, j].conjugate().reshape(1, -1)

    if m is None:
        return U[k, i] * U[:, j].conjugate().reshape(1, -1)

    if k is None:
        return U[:, i].reshape(-1, 1) * U[m, j].conjugate()

    return U[k, i] * U[m, j].conjugate()


@functools.cache
def count_nodes(shape, ex=None):
    if shape[0] != shape[1]:
        raise ValueError(f"expected square matrix, got: {shape}")

    n = np.round(np.log2(shape[0]), 1)

    if ex is not None:
        for i in range(int(n), shape[0]):
            if shape[0] == len(ComputationBasis(i, excitations=ex)):
                return i

    elif abs(n % int(n)) < 1e-15:
        return int(n)

    raise ValueError("could not calculate number of spins")


# @nb.njit
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


def get_diagonal_blocks_edges(elements: set[tuple[int, int]], upperright=True):
    """
    Return horizontal and vertical slices for each block from
    non zero elements position in matrix.
    """
    n_rows = max(i for i, _ in elements)
    n_cols = max(j for _, j in elements)

    blocks = []
    row_up_limit, col_left_limit = 0, 0
    while row_up_limit <= n_rows and col_left_limit <= n_cols:
        row_up = min(i for i, _ in elements if i >= row_up_limit)
        col_left = min(j for _, j in elements if j >= col_left_limit)
        if row_up != col_left:
            raise ValueError(f"upperleft corner not on diagonal: {row_up} != {col_left}")
        col_right = max(j for i, j in elements if i == row_up)
        if not upperright:
            row_down = max(i for i, j in elements if j == col_left)
            if row_down != col_right:
                raise ValueError(
                    f"downright corner not on diagonal: {row_down} != {col_right}"
                )
        else:
            row_down = col_right

        blocks.append(
            (
                slice(row_up, row_down + 1 if row_down < n_rows else None),
                slice(col_left, col_right + 1 if col_right < n_cols else None),
            )
        )
        row_up_limit, col_left_limit = row_down + 1, col_right + 1

    return blocks
