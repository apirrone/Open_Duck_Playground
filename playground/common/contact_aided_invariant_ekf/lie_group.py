import numpy as np
from playground.common.contact_aided_invariant_ekf.utils import CARTESIAN_DIM

TOLERANCE = 1e-10
SMALL_ANGLE_APPROXIMATION = 1e-6


def skew(v: np.ndarray) -> np.ndarray:
    """
    Convert a 3D vector to a skew-symmetric matrix 
    If v has shape (3, N), returns (N, 3, 3) batch of skew-symmetric matrices.
    """
    v = np.asarray(v).squeeze()  # Ensure it's at least 1D

    if v.shape == (3,):  
        v = v[:, None]  # Convert to (3,1) for consistent indexing

    if v.shape[0] != 3:
        raise ValueError(f"Input must have shape (3,), (3, N), or (N,3), but got {v.shape}")

    if v.ndim == 1:  # Single 3D vector case
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    if v.shape[1] == 3:  # Handle (N,3) case by transposing
        v = v.T

    N = v.shape[1]  # Number of vectors
    skew_matrices = np.zeros((N, 3, 3))

    skew_matrices[:, 0, 1] = -v[2]  # -z
    skew_matrices[:, 0, 2] = v[1]   # y
    skew_matrices[:, 1, 0] = v[2]   # z
    skew_matrices[:, 1, 2] = -v[0]  # -x
    skew_matrices[:, 2, 0] = -v[1]  # -y
    skew_matrices[:, 2, 1] = v[0]   # x

    return skew_matrices[0] if N == 1 else skew_matrices  # Return (3,3) if input was a single vector


def exp_so3(w: np.ndarray) -> np.ndarray:
    """ Computes the vectorized exponential map for SO(3) """
    A = skew(w)
    theta = np.linalg.norm(w)
    if theta < TOLERANCE:
        return np.eye(CARTESIAN_DIM)
    elif theta < SMALL_ANGLE_APPROXIMATION:
        return np.eye(CARTESIAN_DIM) + A
    return np.eye(CARTESIAN_DIM) + (np.sin(theta)/theta) * A + ((1 - np.cos(theta)) / (theta**2)) * np.dot(A, A)


def exp_sek3(v: np.ndarray) -> np.ndarray:
    """ Computes the vectorized exponential map for SE_K(3) """
    K = (len(v) - CARTESIAN_DIM) // CARTESIAN_DIM
    X = np.eye(CARTESIAN_DIM + K)
    w = v[:CARTESIAN_DIM]
    theta = np.linalg.norm(w)
    identity = np.eye(CARTESIAN_DIM)

    if theta < TOLERANCE:
        R = identity
        Jl = identity
    else:
        A = skew(w)
        theta2 = theta ** 2
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        one_minus_cos_theta2 = (1 - ctheta) / theta2
        A2 = np.dot(A, A)
        R = identity + (stheta / theta) * A + one_minus_cos_theta2 * A2
        Jl = identity + one_minus_cos_theta2 * A + ((theta - stheta) / (theta2 * theta)) * A2

    X[:CARTESIAN_DIM, :CARTESIAN_DIM] = R
    for i in range(K):
        X[:CARTESIAN_DIM, CARTESIAN_DIM + i] = np.dot(Jl, v[CARTESIAN_DIM + CARTESIAN_DIM * i: 2*CARTESIAN_DIM + CARTESIAN_DIM * i])

    return X


def adjoint_sek3(X: np.ndarray) -> np.ndarray:
    """ Compute Adjoint(X) for X in SE_K(3) """
    K = X.shape[1] - CARTESIAN_DIM
    Adj = np.zeros((CARTESIAN_DIM + CARTESIAN_DIM * K, CARTESIAN_DIM + CARTESIAN_DIM * K))
    R = X[:CARTESIAN_DIM, :CARTESIAN_DIM]
    Adj[:CARTESIAN_DIM, :CARTESIAN_DIM] = R

    for i in range(K):
        Adj[CARTESIAN_DIM + CARTESIAN_DIM * i: 6 + CARTESIAN_DIM * i, CARTESIAN_DIM + CARTESIAN_DIM * i: 6 + CARTESIAN_DIM * i] = R
        Adj[CARTESIAN_DIM + CARTESIAN_DIM * i: 6 + CARTESIAN_DIM * i, :CARTESIAN_DIM] = np.dot(skew(X[:CARTESIAN_DIM, CARTESIAN_DIM + i]), R)

    return Adj