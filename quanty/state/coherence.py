import itertools
from collections import OrderedDict

import numpy as np
import sympy as sp

from quanty import matrix
from quanty.basis import ComputationBasis


def coherence_matrix_elements(order, basis, upperright=True):
    variables = []
    for iv, jv in itertools.combinations_with_replacement(basis, 2):
        order_ = abs(iv.excitations - jv.excitations)
        if order_ != order:
            continue
        i = basis.index(iv)
        j = basis.index(jv)
        variables.append((i, j, True))
        if i != j:
            variables.append((i, j, False))
    return variables


def coherence_matrix(order, basis, var=("x", "y"), dtype=sp.sympify):
    mat = matrix.zeros(len(basis), dtype=dtype)
    variables = []
    for i, j, real in coherence_matrix_elements(order, basis):
        x = sp.Symbol(f"{var[0]}{i}{j}", real=True)
        if real:
            mat[i, j] = x
            variables.extend([(x, (i, j))])
        else:
            y = sp.Symbol(f"{var[1]}{i}{j}", real=True)
            mat[i, j] = x + sp.I * y
            mat[j, i] = x - sp.I * y
            variables.extend([(x, (i, j)), (y, (i, j))])

    return OrderedDict(variables), mat


def assert_coherence_matrix_match(order, basis, mat, decimal=14):
    indecies = {(i, j) for i, j, _ in coherence_matrix_elements(order, basis)}
    residual_max = max(
        np.abs(mat[i, j])
        for i, j in itertools.product(range(len(basis)), range(len(basis)))
        if (i, j) not in indecies and (j, i) not in indecies
    )
    np.testing.assert_almost_equal(residual_max, 0, decimal=decimal)


def coherence_matrix_unlinearize(order, basis, params, ignore=None, dtype=np.ndarray):
    mat = matrix.zeros(len(basis), dtype=dtype)
    params = list(reversed(params))
    for i, j, real in coherence_matrix_elements(order, basis):
        if ignore is not None and ignore(i, j):
            continue
        p = params.pop()
        if real:
            mat[i, j] += p
            if i != j:
                mat[j, i] += p
        else:
            mat[i, j] += 1j * p
            mat[j, i] += -1j * p

    return mat
