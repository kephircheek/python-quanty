import functools
import itertools
import warnings
from collections import OrderedDict

import numpy as np


class BaseVector:
    @classmethod
    def from_str(cls, vector: str):  # i_base_vector
        return cls(int(vector, 2), len(vector))

    def __init__(self, vector: int, n: int):
        if vector < 0 or vector >= 2**n:
            raise ValueError("number should be non-negative and less than 2^n")

        self._v = vector
        self._n = n

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self}>"

    def __len__(self):
        return self._n

    def __int__(self):
        return self._v

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self._v == other._v and self._n == other._n

    def append(self, other):
        return BaseVector.from_str(str(self) + str(other))

    def __str__(self):  # base_vector
        """
        Return `i`-th base vecotor  in full basis set in system with `n` particles.

        NB! Numeration starts from one.

        Examples
        --------
        >>> str(BaseVector(2, 3))
        '010'
        """
        return bin(self._v)[2:].zfill(self._n)

    def insert(self, vector, on: set):
        full_vector = str(self)
        for pos, vector in zip(sorted(list(on)), str(vector)):
            full_vector = full_vector[:pos] + str(vector) + full_vector[pos:]
        return self.__class__.from_str(full_vector)

    @property
    def excitations(self):
        return sum(map(int, str(self)))


class ComputationBasis:
    def __init__(
        self, n: int, excitations: int = None, vectors: list[BaseVector | int] = None
    ):
        self._n = n
        self._ex = excitations
        self.__vectors = (
            None
            if vectors is None
            else OrderedDict((v, i) for i, v in enumerate(vectors))
        )

    @property
    def _vectors(self):
        if self.__vectors is None and self._ex is not None:
            return computation_basis_enumerated(self._n, self._ex)
        return self.__vectors

    @property
    def vectors(self):
        return tuple(self._vectors)

    def copy(self):
        return ComputationBasis(n=self.n, excitations=self.ex, vectors=self.vectors)

    @property
    def n(self):
        return self._n

    @property
    def excitations(self):
        return self._ex or self._n

    @property
    def ex(self):
        return self.excitations

    def __iter__(self):
        if self._vectors is not None:
            return iter(self._vectors.keys())
        return (BaseVector(i, self._n) for i in range(2**self._n))

    def __len__(self):
        if self._vectors is not None:
            return len(self._vectors)
        return 2**self._n

    def index(self, vector):  # reindex
        if self._ex is None:
            return int(vector)
        return self._vectors[vector]

    def sorted(self, key):
        vectors = sorted(self.vectors, key=key)
        return self.__class__(self._n, self._ex, vectors=vectors)

    def sorted_by_excitation(self):
        return self.sorted(key=lambda v: (v.excitations, int(v)))

    def reorder(self, rho, basis):
        import quanty.matrix

        rho_ = quanty.matrix.zeros_like(rho)
        for (ir, r), (ic, c) in itertools.product(
            enumerate(self._vectors), enumerate(self._vectors)
        ):
            ir_, ic_ = basis.index(r), basis.index(c)
            rho_[ir_, ic_] = rho[ir, ic]

        return rho_

    @classmethod
    def reorder_(cls, rho, basis):
        return cls(basis.n, basis.ex).reorder(rho, basis)

    def reversed(self):  # like reverse_order
        """Reverse spin sequence."""
        vectors = (BaseVector.from_str(str(v)[::-1]) for v in self._vectors)
        return ComputationBasis(self._n, self._ex, vectors=vectors)


@functools.lru_cache()
def combination(n, ex):
    # Maybe math.comb ?
    position_combinations = itertools.combinations(range(n), ex)
    return sorted(sum(2**pow for pow in pos) for pos in position_combinations)


@functools.lru_cache(maxsize=1024)
def computation_basis(n, ex=None, only=False):
    """
    Parameters
    ----------
    n (int):
    ex (int): max excitation order
    only (bool): select only setted excitation

    Returns
    -------
        list: of base vectors.

    Examples
    --------
    >>> computation_basis(2)
    (0, 1, 2, 3)
    >>> computation_basis(4, 1)
    (0, 1, 2, 4, 8)
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

    raise ValueError("excitation order should be positive integer or zero")


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

@functools.lru_cache(maxsize=None)
def computation_basis_enumerated(n, ex=None, only=False):
    return OrderedDict(
        (BaseVector(b, n), i)
        for i, b in enumerate(computation_basis(n=n, ex=ex))
    )
