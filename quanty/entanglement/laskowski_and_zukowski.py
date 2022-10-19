"""W. Laskowski and M. Żukowski, Phys. Rev. A 72, 062112 (2005)."""
import numpy as np

from quanty import matrix

__all__ = [
    "k_separable",
    "k_entangled",
]


def max_antidioganal(m: np.ndarray) -> float:
    return float(np.max(np.diag(np.flipud(m))))


def k_separable(m: np.ndarray) -> int:
    r = max_antidioganal(m)
    if r == 0:
        raise ValueError("all antidiagonal elements are zero.")

    n = matrix.count_nodes(m.shape)
    for k in range(1, n + 1):
        if (1 / 2) ** k >= r and r > (1 / 2) ** (k + 1):
            return k

    raise ValueError(f"antidiagonal element too small: {r} <= (1/2)**{n + 1}")


def k_entangled(m: np.ndarray) -> int:
    n = matrix.count_nodes(m.shape)
    k = k_separable(m)
    return int(n // k + (1 if n % k else 0))
