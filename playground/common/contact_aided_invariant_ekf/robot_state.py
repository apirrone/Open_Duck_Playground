import numpy as np
from typing import Optional
from playground.common.contact_aided_invariant_ekf.utils import CARTESIAN_DIM, RIGID_TRANSFORM_SIZE


def set_array_w_different_shape(
    array_to_set: np.ndarray,
    array_w_different_shape: np.ndarray
):
    return np.reshape(array_w_different_shape, array_to_set.shape)


class RobotState:
    """
    Robot state
    X = [R_WB, v_B, p_WB, p_C1, p_C2]
    P = Covariance
    """
    def __init__(
            self,
            X: Optional[np.ndarray] = None,
            Theta: Optional[np.ndarray] = None,
            P: Optional[np.ndarray] = None
    ):
        if X is None:
            X = np.eye(5)
        if Theta is None:
            Theta = np.zeros((2*CARTESIAN_DIM, 1))
        
        self.X_ = X
        self.Theta_ = Theta
        
        if P is None:
            P = np.eye(CARTESIAN_DIM * self.dim_x() + self.dim_theta() - 2*CARTESIAN_DIM)

        self.P_ = P
    
    def get_x(self) -> np.ndarray:
        return self.X_.copy()

    def get_theta(self) -> np.ndarray:
        return self.Theta_.copy()

    def get_p(self) -> np.ndarray:
        return self.P_.copy()

    def get_rotation(self) -> np.ndarray:
        return self.X_[:CARTESIAN_DIM, :CARTESIAN_DIM].copy()

    def get_velocity(self) -> np.ndarray:
        return self.X_[:CARTESIAN_DIM, CARTESIAN_DIM].copy()

    def get_position(self) -> np.ndarray:
        return self.X_[:CARTESIAN_DIM, RIGID_TRANSFORM_SIZE].copy()

    def get_gyroscope_bias(self) -> np.ndarray:
        return self.Theta_[:CARTESIAN_DIM].copy()

    def get_accelerometer_bias(self) -> np.ndarray:
        return self.Theta_[-CARTESIAN_DIM:].copy()

    def dim_x(self) -> np.ndarray:
        return self.X_.shape[1]

    def dim_theta(self) -> np.ndarray:
        return self.Theta_.shape[0]

    def dim_p(self) -> np.ndarray:
        return self.P_.shape[1]

    def set_x(self, X: np.ndarray):
        self.X_ = X

    def set_theta(self, Theta: np.ndarray):
        self.Theta_ = Theta

    def set_p(self, P: np.ndarray):
        self.P_ = P

    def set_rotation(self, R: np.ndarray):
        self.X_[:CARTESIAN_DIM, :CARTESIAN_DIM] = R.copy()

    def set_velocity(self, v: np.ndarray):
        self.X_[:CARTESIAN_DIM, CARTESIAN_DIM] = v.reshape(
            self.X_[:CARTESIAN_DIM, CARTESIAN_DIM].shape
        ).copy()

    def set_position(self, p: np.ndarray):
        self.X_[:CARTESIAN_DIM, CARTESIAN_DIM + 1] = p.reshape(
            self.X_[:CARTESIAN_DIM, CARTESIAN_DIM + 1].shape
        ).copy()

    def set_gyroscope_bias(self, bg: np.ndarray):
        bg = self.Theta_[:CARTESIAN_DIM] = bg.reshape(
            self.Theta_[:CARTESIAN_DIM].shape
        )

    def set_accelerometer_bias(self, ba: np.ndarray):
        ba = ba.reshape(
            self.Theta_[-CARTESIAN_DIM:].shape
        )

    def copy_diag_x(self, n):
        """
        Create a block diagonal matrix with X as the diagonal elements.
        """
        dimX = self.dim_x()

        # Create a list of n identity blocks of size (dimX, dimX)
        blocks = [np.eye(dimX) for _ in range(n)]

        # Create a list of `X_` blocks along the diagonal
        blocks = [self.X_ if i == j else np.zeros((dimX, dimX)) for i in range(n) for j in range(n)]

        # Reshape into a (n, n) structure and use np.block to assemble the matrix
        BigX = np.block([[blocks[i * n + j] for j in range(n)] for i in range(n)])
    
        return BigX

    def __str__(self) -> str:
        return ("--------- Robot State -------------\n"
                    f"X:\n{self.X_}\n\n"
                    f"Theta:\n{self.Theta_}\n\n"
                    f"P:\n{self.P_}\n"
                    "-----------------------------------")