import numpy as np
from typing import Callable


class Kinematics:
    def __init__(self, id_in: int, pose_in: np.ndarray, covariance_in: np.ndarray):
        """
        Kinematics class to store kinematic measurements and their covariance.

        Parameters:
        - id_in (int): Identifier for the measurement.
        - pose_in (np.ndarray): 4x4 transformation matrix computed by Kinematics.
        - covariance_in (np.ndarray): 6x6 covariance matrix computed by the corresponding Kinematic Jacobian.
        """
        assert pose_in.shape == (4, 4), "Pose must be a 4x4 matrix."
        assert covariance_in.shape == (6, 6), "Covariance must be a 6x6 matrix."
        
        self.id = id_in
        self.pose = pose_in
        self.covariance = covariance_in


def compute_kinematics_covariance(
    joint_positions: np.ndarray,
    contact_cov: np.ndarray,
    body_jacobian_functor: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Note: errors have been defined with right-invariance (meaning body expressed in the body frame). Therefore, we need the body Jacobian of SE(3).
    """
    body_jacobian = body_jacobian_functor(joint_positions)
    return body_jacobian @ contact_cov @ body_jacobian.T