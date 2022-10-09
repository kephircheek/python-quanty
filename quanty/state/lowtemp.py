import itertools

import numpy as np

from quanty import matrix
from quanty.basis import BaseVector, ComputationBasis


def init_low_temp_chain(sender_rho, basis, sender_basis=None, dtype=np.complex128):
    n_sender = matrix.count_nodes(sender_rho.shape, ex=basis.ex)
    sender_basis = sender_basis or ComputationBasis(matrix.count_nodes(sender_rho.shape))
    rho_init = matrix.zeros(len(basis), dtype=dtype)
    for i_state, j_state in itertools.product(sender_basis, sender_basis):
        i, j = sender_basis.index(i_state), sender_basis.index(j_state)
        ground_state = BaseVector(0, basis.n - n_sender)
        irow = basis.index(i_state.append(ground_state))
        icol = basis.index(j_state.append(ground_state))
        rho_init[irow, icol] = sender_rho[i, j]
        if irow != icol:
            rho_init[icol, irow] = sender_rho[j, i]

    return rho_init
