import numpy as np
from scipy import sparse

SIGMA_X = [[0, 1], [1, 0]]
SIGMA_Y = [[0, -1], [1, 0]]
SIGMA_Z = [[1, 0], [0, -1]]
SIGMA_PLUS = [[0, 1], [0, 0]]
SIGMA_MINUS = [[0, 0], [1, 0]]


def eye(n=1, dtype=None):
    """eye matrix

    Examples:
        >>> eye().toarray()
        array([[1, 0],
               [0, 1]], dtype=int16)

        >>> eye(2).toarray()
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], dtype=int16)

    """
    dtype = dtype or np.int16
    return sparse.eye(2**n, dtype=dtype)


def zeros(n, dtype=None):
    dtype = dtype or np.int16
    return eye(n, dtype) - eye(n, dtype)


def _sigma(sigma, n=1, k=None, dtype=None):
    if k is not None and k >= n:
        raise ValueError(f"spin number too high: {k} >= {n}")

    dtype = dtype or np.int16

    if k is None and n == 1:
        return sparse.csc_matrix(np.array(sigma, dtype=dtype))

    if k is not None:
        return sparse.kron(
            sparse.kron(
                eye(k),
                _sigma(sigma, n=1),
                format="csc",
            ),
            eye(n - k - 1),
            format="csc",
        )

    return sum((_sigma(sigma, n, i, dtype=dtype) for i in range(n)), zeros(n))


def sx(n=1, k=None, dtype=None):
    """Projector on x axis

    Example:
        >>> sx(2).toarray()
        array([[0, 1, 1, 0],
               [1, 0, 0, 1],
               [1, 0, 0, 1],
               [0, 1, 1, 0]], dtype=int16)

        >>> sx(1).toarray()
        array([[0, 1],
               [1, 0]], dtype=int16)
    """
    return _sigma(SIGMA_X, n=n, k=k, dtype=dtype)


def sy(n=1, k=None, dtype=None):
    """Projector on y axis

    Example:
        >>> sy().toarray()
        array([[ 0, -1],
               [ 1,  0]], dtype=int16)

        >>> sy(2).toarray()
        array([[ 0, -1, -1,  0],
               [ 1,  0,  0, -1],
               [ 1,  0,  0, -1],
               [ 0,  1,  1,  0]], dtype=int16)

        >>> sy(2, 0).toarray()
        array([[ 0,  0, -1,  0],
               [ 0,  0,  0, -1],
               [ 1,  0,  0,  0],
               [ 0,  1,  0,  0]], dtype=int16)

    """
    return _sigma(SIGMA_Y, n=n, k=k, dtype=dtype)


def sz(n=1, k=None, dtype=None):
    """Projector on z-axis

    Examples:
        >>> sz().toarray()
        array([[ 1,  0],
               [ 0, -1]], dtype=int16)

        >>> sz(2, 0).toarray()
        array([[ 1,  0,  0,  0],
               [ 0,  1,  0,  0],
               [ 0,  0, -1,  0],
               [ 0,  0,  0, -1]], dtype=int16)

        >>> sz(2, 1).toarray()
        array([[ 1,  0,  0,  0],
               [ 0, -1,  0,  0],
               [ 0,  0,  1,  0],
               [ 0,  0,  0, -1]], dtype=int16)

        >>> sz(2).toarray()
        array([[ 2,  0,  0,  0],
               [ 0,  0,  0,  0],
               [ 0,  0,  0,  0],
               [ 0,  0,  0, -2]], dtype=int16)

    """

    return _sigma(SIGMA_Z, n=n, k=k, dtype=dtype)


def sup(n=1, k=None, dtype=None):
    """

    Example:
        >>> sup().toarray()
        array([[0, 1],
               [0, 0]], dtype=int16)

        >>> (sup(1) @ sup(1)).toarray()
        array([[0, 0],
               [0, 0]], dtype=int16)
    """
    return _sigma(SIGMA_PLUS, n=n, k=k, dtype=dtype)


def sdown(n=1, k=None, dtype=None):
    """

    Example:
        >>> sdown().toarray()
        array([[0, 0],
               [1, 0]], dtype=int16)
        >>> (sdown(1) @ sdown(1)).toarray()
        array([[0, 0],
               [0, 0]], dtype=int16)
    """
    return _sigma(SIGMA_MINUS, n=n, k=k, dtype=dtype)


def isempty(mat):
    """check matrix with only zero values

    Example:
        >>> isempty(sz(2) - sz(2))
        True

        >>> isempty(sx(2) - sy(2))
        False

    """
    return mat.sum().sum() == 0


def commutator(left, right):
    """commutator of two matrix [A, B] = A * B - B * A

    Example:
        >>> isempty(commutator(sx(), sy()) - sz())
        True

        >>> isempty(commutator(sy(), sz()))
        False

    """
    return left @ right - right @ left


def trace(mat):
    """trace of matrix

    Example:
        >>> trace(sz(5))
        0
        >>> trace(eye(5))
        32
    """

    return mat.diagonal().sum()


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
