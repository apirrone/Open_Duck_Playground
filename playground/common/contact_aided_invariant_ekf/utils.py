import numpy as np
CARTESIAN_DIM = 3
RIGID_TRANSFORM_SIZE = 4
GRAVITY_ACCELERATION = -9.81


def remove_row_and_column(M, index):
    # dim_x = M.shape[1]
    # # Print statement for debugging (similar to the C++ code)
    # # print(f"Removing index: {index}")

    # Remove the row
    M = np.delete(M, index, axis=0)

    # Remove the column
    M = np.delete(M, index, axis=1)

    return M
