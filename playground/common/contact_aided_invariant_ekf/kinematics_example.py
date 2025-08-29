import numpy as np
import cProfile
import pstats
from matplotlib import pyplot as plt
from playground.common.contact_aided_invariant_ekf.robot_state import RobotState
from playground.common.contact_aided_invariant_ekf.noise_params import NoiseParams
from playground.common.contact_aided_invariant_ekf.contact_aided_invariant_ekf import RightInEKF, Kinematics

np.set_printoptions(precision=5, linewidth=100)

# Conversion from string to numeric
def stod98(s):
    return float(s)


def stoi98(s):
    return int(stod98(s))


def load_matrices_from_binary(filename):
    """
    Some reason this gets transposed when it comes in.
    """
    matrices = []
    with open(filename, "rb") as f:
        while True:
            timestep_data = f.read(4)
            if not timestep_data:
                break  # EOF
            
            timestep = np.frombuffer(timestep_data, dtype=np.int32)[0]
            rows = np.frombuffer(f.read(4), dtype=np.int32)[0]
            cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
            data = np.frombuffer(f.read(rows * cols * 8), dtype=np.float64)
            
            matrices.append((timestep, data.reshape((rows, cols)).T))
    
    return matrices


# Main function
def main():
    # ---- Initialize invariant extended Kalman filter ----- //
    initial_state = RobotState()

    # Initialize state mean
    R0 = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])
    v0 = np.zeros(3)
    p0 = np.zeros(3)
    bg0 = np.zeros(3)
    ba0 = np.zeros(3)
    initial_state.set_rotation(R0)
    initial_state.set_velocity(v0)
    initial_state.set_position(p0)
    initial_state.set_gyroscope_bias(bg0)
    initial_state.set_accelerometer_bias(ba0)

    # Initialize state covariance
    noise_params = NoiseParams(
        gyroscope_nosie=0.01,
        accelerometer_noise=0.1,
        gyroscope_bias_noise=0.00001,
        accelerometer_bias_noise=0.0001,
        contact_noise=0.01
    )

    # Initialize filter
    filter = RightInEKF(initial_state, noise_params)
    # print("Noise parameters are initialized to:")
    # print(filter.get_noise_params())
    # print("Robot's state is initialized to:")
    # print(filter.get_state())

    # Open data file
    data_file = "./playground/common/contact_aided_invariant_ekf/data/imu_kinematic_measurements.txt"
    validation_data_file = "./playground/common/contact_aided_invariant_ekf/data/robot_state_estimates.bin"
    kinematic_covariance_data_file = "./playground/common/contact_aided_invariant_ekf/data/kinematic_covariance_updates.bin"
    prop_covariance_data_file = "./playground/common/contact_aided_invariant_ekf/data/propagation_updates.bin"
    imu_measurement_prev = np.zeros(6)
    t_prev = 0
    count = 0
    kinematic_count = 0
    prop_count = 0

    matrices = load_matrices_from_binary(validation_data_file)
    kinematic_matrices = load_matrices_from_binary(kinematic_covariance_data_file)
    propagation_matrices = load_matrices_from_binary(prop_covariance_data_file)
    # ---- Loop through data file and read in measurements line by line ---- //
    # TODO: debug propagation step at count 636. Seems like after adding a contact, it doesn't update properly.
    error_list = []
    contact_list = []
    propagation_computation_time_list = []
    with open(data_file, 'r') as infile:
        for line in infile:
            measurement = line.split()
            # print(f"Count: {count}")
            # # print("X: ", filter.get_state())
            actual_state = filter.get_state()
            # Load and print matrices
            adjusted_idx = count * 3
            validation_X = matrices[adjusted_idx][1]
            validation_Theta = matrices[adjusted_idx + 1][1]
            validation_P = matrices[adjusted_idx + 2][1]

            # # print("Validation X: ", validation_X)
            
            error_X = actual_state.get_x() - validation_X
            error_P = actual_state.get_p() - validation_P
            # # print("Error X: ", error_X)
            # # print("Error P: ", error_P)
            
            norm_error_X = np.linalg.norm(error_X)
            norm_error_P = np.linalg.norm(error_P, axis=1)
            # print("Norm Error X: ", norm_error_X)
            # print("Norm Error P: ", norm_error_P)
            error_list.append(norm_error_X)
            count += 1

            if measurement[0] == "IMU":
                # print("Received IMU Data, propagating state")
                assert len(measurement) - 2 == 6
                t = stod98(measurement[1])
                imu_measurement = np.array([stod98(x) for x in measurement[2:8]])

                # Propagate using IMU data
                dt = t - t_prev
                if 1e-6 < dt < 1:
                    filter.propagate(imu_measurement_prev, dt)
                    # propagation_computation_time_list.append()
            elif measurement[0] == "CONTACT":
                # print("Received CONTACT Data, setting filter's contact state")
                assert (len(measurement) - 2) % 2 == 0
                contacts = []
                t = stod98(measurement[1])
                for i in range(2, len(measurement), 2):
                    id = stoi98(measurement[i])
                    indicator = bool(stod98(measurement[i + 1]))
                    if indicator:
                        pass
                    contacts.append((id, indicator))
                filter.set_contacts(contacts)
                contact_list.append((count, contacts[0][1], contacts[1][1]))
                # # print("Set contacts: ",filter.contacts_)
            elif measurement[0] == "KINEMATIC":
                # print("Received KINEMATIC observation, correcting state")
                assert (len(measurement) - 2) % 44 == 0
                measured_kinematics = []
                t = stod98(measurement[1])
                # TODO: need to do the computations for Jacobians and covariance computations from them.
                for i in range(2, len(measurement), 44):
                    id = stoi98(measurement[i])
                    q = np.array([stod98(measurement[i+j]) for j in range(1, 5)])
                    q = q / np.linalg.norm(q)  # Normalize quaternion
                    p = np.array([stod98(measurement[i + 5]), stod98(measurement[i + 6]), stod98(measurement[i + 7])])
                    pose = np.eye(4)
                    pose[:3, :3] = q_to_rotation_matrix(q)  # Assuming a function q_to_rotation_matrix exists
                    pose[:3, 3] = p
                    covariance = np.array([[stod98(measurement[i + 8 + j * 6 + k]) for k in range(6)] for j in range(6)])
                    # print("Kinematic pose: ", pose)
                    # print("Kinematic cov: ", covariance)
                    measured_kinematics.append(
                        Kinematics(id_in=id, pose_in=pose, covariance_in=covariance)
                        )
                filter.correct_kinematics(measured_kinematics)

            # Store previous timestamp
            t_prev = t
            imu_measurement_prev = imu_measurement
    error_array = np.array(error_list)
    contact_count_array = np.array([contact_tuple[0] for contact_tuple in contact_list])
    contact_array = np.array([[contact_tuple[1], contact_tuple[2]] for contact_tuple in contact_list], dtype=float)
    contact_array *= np.max(error_array) / 2
    # plt.plot(error_list, label="State Error")
    # plt.plot(contact_count_array, contact_array[:,0], label="Contact 0")
    # plt.plot(contact_count_array, contact_array[:,1], label="Contact 1")
    # plt.legend()
    # plt.show()
    # Print final state
    print(filter.get_state())


def q_to_rotation_matrix(q):
    # TODO: use scipy?
    # Assuming q is a normalized quaternion
    w, x, y, z = q
    return np.array([[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2]])


if __name__ == "__main__":
    main()
    # cProfile.run("main()")

    # stats = pstats.Stats(pr)
    # stats.strip_dirs().sort_stats("cumulative").print_stats(20)  # Top 20 cumulative time 
    
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # pr.print_stats(sort="cumulative").print_stats(10)