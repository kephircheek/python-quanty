import itertools

import numpy as np

from quanty.basis import BaseVector, ComputationBasis
from quanty.matrix import Type, zeros

from .abc import Hamiltonian


class XX(Hamiltonian):
    def __call__(self, n: int, ex: int = None):
        return _xx_pure(
            n=n,
            dc=self.model.constant,
            ex=ex,
        )


def _xx_qutip(n, dc):

    import qutip as qp
    from qutip import qeye
    from qutip import sigmax as sx
    from qutip import sigmay as sy
    from qutip import sigmaz as sz
    from qutip import tensor

    ham = 0
    for i, j in itertools.combinations(range(n), 2):
        xx_, yy_ = [qp.qeye(2)] * n, [qp.qeye(2)] * n
        xx_[i], xx_[j] = qp.sigmax(), qp.sigmax()
        yy_[i], yy_[j] = qp.sigmay(), qp.sigmay()
        ham += dc(i, j) * (tensor(*xx_) + tensor(*yy_))

    return ham / 4


def _xx_pure(n, dc=None, ex=None, dtype=np.float64):
    """
    $ H_{XX} = \sum_{ij} D_{ij} \left( I_{xi}I_{xj} +  I_{xi}I_{xj} \right) $
    """
    basis = ComputationBasis(n, ex)
    ham = zeros(len(basis), dtype=dtype)

    for i, j in itertools.combinations(range(n), 2):

        dc_ = dc(i, j)
        if dc_ == 0 or dc_ is None:
            continue

        n_remain = n - 2
        basis_remain = []
        if ex is None:
            basis_remain = ComputationBasis(n_remain)
        elif (_ex := ex - 1) >= 0:
            basis_remain = ComputationBasis(n_remain, excitations=_ex)

        # only |...0...1...><...1...0...| non zero element of I^+I^-
        indices = (
            (
                basis.index(ivec.insert(BaseVector.from_str("01"), {i, j})),
                basis.index(ivec.insert(BaseVector.from_str("10"), {i, j})),
            )
            for ivec in basis_remain
        )
        for irow, icol in indices:
            ham[irow, icol] = dc_ / 2
            ham[icol, irow] = dc_ / 2

    return ham
