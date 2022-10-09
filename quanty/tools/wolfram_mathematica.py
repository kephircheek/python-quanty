import numpy as np


def to_mathematica(matrix: list):
    return str(matrix).replace("[", "{").replace("]", "}").replace("j", "*I")


def from_mathematica(data):
    return np.array(
        eval(
            data.replace("{", "[").replace("}", "]").replace("I", "1j").replace("*^", "e")
        )
    )
