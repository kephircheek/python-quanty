import functools

import numpy as np
import sympy as sp

from quanty.matrix import Type


class Hamiltonian:
    def __init__(self, model, dtype=None):
        self._model = model
        self._dtype = dtype  # or Type.NUMPY_NDARRAY

    @property
    def model(self):
        return self._model

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, n: int, ex: int = None):
        raise NotImplementedError

    @functools.lru_cache(maxsize=1024)
    def eigh(self, n: int, ex: int = None):
        """
        Returns:
        --------
        np.ndarray: eigvals
        np.ndarray: eigvecs
        """
        if self.dtype is None:  # == Type.NUMPY_NDARRAY:
            return np.linalg.eigh(self(n, ex=ex))

        elif self.dtype == Type.SYMPY_MATRIX:
            # can use sp.Matrix.diagonalize() ?
            eigenvals, eigenvecs = [], []
            for eigenval, gmult, subeigvecs in self(n).eigenvects():
                eigenvals.extend([eigenval] * gmult)
                eigenvecs.extend([vec.normalized() for vec in subeigvecs])

            return sp.Matrix(eigenvals), sp.Matrix.hstack(*eigenvecs)
        else:
            raise TypeError(f"unsupported type: {self.dtype}")

    @functools.lru_cache(maxsize=1024)
    def U(self, n: int, dt: float, ex: int = None):
        if self.dtype is None:  # == Type.NUMPY_NDARRAY:
            eigvals, eigvecs = self.eigh(n, ex=ex)
            eigvals = eigvals.astype(np.complex128)
            return eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
        else:
            raise TypeError(f"unsupported type: {self.dtype}")
