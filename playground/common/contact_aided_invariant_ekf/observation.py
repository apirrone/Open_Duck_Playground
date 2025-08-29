import numpy as np


def check_measurement_is_empty(Y: np.ndarray):
    return Y.shape[0] == 0


class Observation:
    def __init__(
            self, Y: np.ndarray,
            b: np.ndarray,
            H: np.ndarray,
            N: np.ndarray,
            PI: np.ndarray
    ):
        self.Y = Y
        self.b = b
        self.H = H
        self.N = N
        self.PI = PI

    def is_empty(self):
        return check_measurement_is_empty(self.Y)

    def __str__(self):
        return ("---------- Observation ------------\n"
                f"Y:\n{self.Y}\n\n"
                f"b:\n{self.b}\n\n"
                f"H:\n{self.H}\n\n"
                f"N:\n{self.N}\n\n"
                f"PI:\n{self.PI}\n"
                "-----------------------------------")