from .abc import Hamiltonian
from .xx import _xx_pure
from .zz import _zz_pure


class XXZ(Hamiltonian):
    def __call__(self, n, ex: int = None):
        return _xxz_pure(n, self.model.constant, ex=ex)


def _xxz_pure(n, dc=None, ex=None):
    return _xx_pure(n, dc, ex) - 2 * _zz_pure(n, dc, ex)
