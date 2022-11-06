import itertools
from collections import OrderedDict

import sympy as sp
import numpy as np

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


def coherence_matrix_unlinearize(order, basis, params, dtype=np.ndarray):
    mat = matrix.zeros(len(basis), dtype=dtype)
    params = list(reversed(params))
    for i, j, real in coherence_matrix_elements(order, basis):
        p = params.pop()
        if real:
            mat[i, j] += p
            if i != j:
                mat[j, i] += p
        else:
            mat[i, j] += 1j * p
            mat[j, i] += -1j * p

    return mat
