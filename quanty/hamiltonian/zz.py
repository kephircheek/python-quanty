import itertools

import numpy as np

from quanty.basis import BaseVector, ComputationBasis
from quanty.matrix import zeros


def _zz_pure(n, dc=None, ex=None, dtype=np.float64):
    """
    $ H_{ZZ} = \sum_{ij} D_{ij} I_{zi}I_{zj} $
    """
    basis = ComputationBasis(n, ex)
    ham = zeros(len(basis), dtype=dtype)

    for i, j in itertools.combinations(range(n), 2):

        dc_ = dc(i, j)
        if dc_ == 0 or dc_ is None:
            continue

        for sign, ex_, vector in [
            (1, 0, "00"),
            (-1, 1, "01"),
            (-1, 1, "10"),
            (1, 2, "11"),
        ]:
            indices = (
                (
                    basis.index(ivec.insert(BaseVector.from_str(vector), {i, j})),
                    basis.index(ivec.insert(BaseVector.from_str(vector), {i, j})),
                )
                for ivec in ComputationBasis(
                    n - 2, excitations=((ex - ex_) if ex else None)
                )
            )
            for irow, icol in indices:
                ham[irow, icol] += sign * dc_ / 4

    return ham
