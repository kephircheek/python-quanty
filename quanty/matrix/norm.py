import numpy as np


def frobenius_norm(mat):
    return np.sum(np.abs(mat) ** 2)
