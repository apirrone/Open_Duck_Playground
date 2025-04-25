import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as sp_R
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

from playground.common.contact_aided_invariant_ekf.contact_aided_invariant_ekf import (
    RightInEKF,
)
from playground.common.contact_aided_invariant_ekf.robot_state import RobotState
from playground.common.contact_aided_invariant_ekf.noise_params import NoiseParams
from playground.common.contact_aided_invariant_ekf.drake_kinematics import DrakeKinematics

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool, urdf_model_path: str
    ):
        super().__init__(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

        # Gather body and feet id
        self.body_id = self.data.body("imu").id
        self.feet_ids = {
            0: self.data.body("foot_assembly").id,
            1: self.data.body("foot_assembly_2").id,
        }

        pose = self.get_pose_from_id(self.body_id)

        # Convert to rotation matrix
        body_rotation_matrix = pose[0:3, 0:3]
        self.ekf_noise_params = NoiseParams() # TODO: put in real
        # self.ekf_noise_params.set_gyroscope_bias_noise(1e-6)
        # ekf_noise_params.set_contact_noise(0.5)
        ekf_robot_state = RobotState()
        ekf_robot_state.set_position(pose[0:3, 3])
        ekf_robot_state.set_rotation(body_rotation_matrix)
        ekf_robot_state.set_velocity(np.zeros(3))
        self._ekf = RightInEKF(ekf_robot_state, self.ekf_noise_params)

        # Define the forward kinematics source.
        self.drake_kinematics = DrakeKinematics(
            imu_frame_name="imu",
            urdf_model_path=urdf_model_path,
            end_effector_frame_name_to_id_map={
                "foot_assembly": 0,
                "foot_assembly_2": 1,
            },
            # From URDF assembly to foot. And then add additional padding from foot link to bottom of foot.
            end_effector_offset_map={
                "foot_assembly": [0.0005, -0.036225 - 0.008, 0.01955],
                "foot_assembly_2": [0.0005, -0.036225 - 0.008, 0.01955],
            },
        )

        # Define the mapping from mujoco index to drake index
        self.mask = np.zeros(
            len(self.drake_kinematics.joint_name_to_idx_map), dtype=int
        )
        for name, idx in self.drake_kinematics.joint_name_to_idx_map.items():
            self.mask[idx] = self.actuator_name_to_idx_map[name]

        covariance_scaling_factor = 10.0 # Just using a random scaling term to sample initial biases. Then these covariances are used to do a random walk.
        self.gyro_bias = np.random.multivariate_normal(np.zeros(3), covariance_scaling_factor*self.ekf_noise_params.get_gyroscope_bias_cov())
        self.accelerometer_bias = np.random.multivariate_normal(np.zeros(3), covariance_scaling_factor*self.ekf_noise_params.get_accelerometer_bias_cov())

        # grab initial sim time
        self.prev_sim_time = self.data.time

        # Variables for logging.
        self.ekf_imu_measurement_prev = np.zeros(6)
        self.timestamps = []

        # Lists for true values
        self.true_rotations = []
        self.true_positions = []
        self.true_velocities = []

        # Lists for estimated values
        self.est_rotations = []
        self.est_positions = []
        self.est_velocities = []

        self.true_gyro_bias = []
        self.true_accelerometer_bias = []

        self.est_gyro_bias = []
        self.est_accelerometer_bias = []

        self.timestamps = []
        self.current_time = 0  # We'll just use 1

    def get_pose_from_id(self, id: str) -> np.ndarray:
        pos = self.data.xpos[id]
        quat = self.data.xquat[id]
        pose = np.eye(4)
        pose[:3, :3] = sp_R.from_quat(np.roll(quat, -1)).as_matrix()
        pose[:3, 3] = pos
        return pose

    def store_values(self, true_rot, true_pos, true_vel, est_rot, est_pos, est_vel, true_gyro_bias, true_accelerometer_bias, est_gyro_bias, est_accelerometer_bias):
        """
        Helper function which stores estimated EKF values and the actual values.
        """
        # Update time
        self.timestamps.append(self.current_time)
        self.current_time += self.sim_dt  # Assuming you have a timestep dt

        # Store true values
        self.true_rotations.append(true_rot.copy())
        self.true_positions.append(true_pos.copy())
        self.true_velocities.append(true_vel.copy())

        # Store estimated values
        self.est_rotations.append(est_rot.copy())
        self.est_positions.append(est_pos.copy())
        self.est_velocities.append(est_vel.copy())

        self.true_gyro_bias.append(true_gyro_bias.copy())
        self.true_accelerometer_bias.append(true_accelerometer_bias.copy())

        self.est_gyro_bias.append(est_gyro_bias.copy())
        self.est_accelerometer_bias.append(est_accelerometer_bias.copy())

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
        correct_kin: bool
    ):
        """
        correct_kin (bool): specifies if we should run the measurement update with the leg kinematics for this timestep.
        """
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        ekf_imu_measurement = np.zeros(6)
        ekf_imu_measurement[0:3] = gyro
        ekf_imu_measurement[3:6] = accelerometer

        body_pose_world = self.get_pose_from_id(self.body_id)

        current_sim_time = data.time
        elapsed_sim_time = current_sim_time - self.prev_sim_time
        self._ekf.propagate(
            self.ekf_imu_measurement_prev,  # use previous measurement.
            dt=elapsed_sim_time,
        )

        # Update prev sim time
        self.prev_sim_time = current_sim_time

        self.ekf_imu_measurement_prev = ekf_imu_measurement.copy()

        ekf_contact_measurement = [(idx, contacts[idx]) for idx in range(len(contacts))]
        self._ekf.set_contacts(ekf_contact_measurement)
        joint_uncertainty = 1e-3
        feet_to_kinematics_map = self.drake_kinematics.compute_kinematics(
            data.qpos[self.mask],
            joint_uncertainty * np.eye(len(joint_angles)),
        )

        if correct_kin:
            self._ekf.correct_kinematics(
                measured_kinematics=list(feet_to_kinematics_map.values())
            )

        estimated_state = self._ekf.get_state()

        floating_base_vel = data.qvel[self._floating_base_qvel_addr:self._floating_base_qvel_addr + 6]
        self.store_values(
            true_rot=sp_R.from_matrix(body_pose_world[0:3, 0:3]).as_euler(
                "xyz", degrees=True
            ),
            true_pos=body_pose_world[0:3, 3],
            true_vel=body_pose_world[0:3, 0:3].T @ floating_base_vel[0:3],
            est_rot=sp_R.from_matrix(estimated_state.get_rotation()).as_euler(
                "xyz", degrees=True
            ),
            est_pos=estimated_state.get_position(),
            est_vel=estimated_state.get_rotation().T @ estimated_state.get_velocity(),
            true_gyro_bias=self.gyro_bias,
            true_accelerometer_bias=self.accelerometer_bias,
            est_gyro_bias=estimated_state.get_gyroscope_bias().squeeze(),
            est_accelerometer_bias=estimated_state.get_gyroscope_bias().squeeze(),
        )

        # How to get lin_vel in body frame
        # lin_vel = estimated_state.get_rotation().T @ estimated_state.get_velocity()

        estimated_pose = np.eye(4)
        estimated_pose[0:3, 0:3] = estimated_state.get_rotation()
        estimated_pose[0:3, 3] = estimated_state.get_position()

        # Playing around with utlizing the path frame from the Disney paper.
        # imu_frame_in_path_frame = np.linalg.inv(self.path_frame) @ estimated_pose

        # Note: I have removed the estimated state from the obs for now, but it is easy to add in.
        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                # gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase,
            ]
        )

        return obs

    def plot_comparisons(self):
        """
        Helper function to plot EKF estimation vs actual.
        """
        # Convert lists to numpy arrays
        true_rotations = np.array(self.true_rotations)
        true_positions = np.array(self.true_positions)
        true_velocities = np.array(self.true_velocities)

        est_rotations = np.array(self.est_rotations)
        est_positions = np.array(self.est_positions)
        est_velocities = np.array(self.est_velocities)

        true_gyro_bias = np.array(self.true_gyro_bias)
        true_accelerometer_bias = np.array(self.true_accelerometer_bias)

        est_gyro_bias = np.array(self.est_gyro_bias)
        est_accelerometer_bias = np.array(self.est_accelerometer_bias)

        timestamps = np.array(self.timestamps)

        # Create figure with three subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 15))

        # Plot rotations
        ax1.plot(timestamps, true_rotations[:, 0], "r-", label="True Roll")
        ax1.plot(timestamps, true_rotations[:, 1], "g-", label="True Pitch")
        ax1.plot(timestamps, true_rotations[:, 2], "b-", label="True Yaw")
        ax1.plot(timestamps, est_rotations[:, 0], "r--", label="Est Roll")
        ax1.plot(timestamps, est_rotations[:, 1], "g--", label="Est Pitch")
        ax1.plot(timestamps, est_rotations[:, 2], "b--", label="Est Yaw")
        ax1.set_ylabel("Rotation (degrees)")
        ax1.set_title("True vs Estimated Rotations")
        ax1.legend()
        ax1.grid(True)

        # Plot positions
        ax2.plot(timestamps, true_positions[:, 0], "r-", label="True X")
        ax2.plot(timestamps, true_positions[:, 1], "g-", label="True Y")
        ax2.plot(timestamps, true_positions[:, 2], "b-", label="True Z")
        ax2.plot(timestamps, est_positions[:, 0], "r--", label="Est X")
        ax2.plot(timestamps, est_positions[:, 1], "g--", label="Est Y")
        ax2.plot(timestamps, est_positions[:, 2], "b--", label="Est Z")
        ax2.set_ylabel("Position (m)")
        ax2.set_title("True vs Estimated Positions")
        ax2.legend()
        ax2.grid(True)

        # Plot velocities
        ax3.plot(timestamps, true_velocities[:, 0], "r-", label="True X")
        ax3.plot(timestamps, true_velocities[:, 1], "g-", label="True Y")
        ax3.plot(timestamps, true_velocities[:, 2], "b-", label="True Z")
        ax3.plot(timestamps, est_velocities[:, 0], "r--", label="Est X")
        ax3.plot(timestamps, est_velocities[:, 1], "g--", label="Est Y")
        ax3.plot(timestamps, est_velocities[:, 2], "b--", label="Est Z")

        ax3.set_ylabel("Velocity (m/s)")
        ax3.set_xlabel("Time (s)")
        ax3.set_title("True vs Estimated Velocities")
        ax3.legend()
        ax3.grid(True)

        # Plot gyro biases
        ax4.plot(timestamps, true_gyro_bias[:, 0], "r-", label="True X")
        ax4.plot(timestamps, true_gyro_bias[:, 1], "g-", label="True Y")
        ax4.plot(timestamps, true_gyro_bias[:, 2], "b-", label="True Z")
        ax4.plot(timestamps, est_gyro_bias[:, 0], "r--", label="Est X")
        ax4.plot(timestamps, est_gyro_bias[:, 1], "g--", label="Est Y")
        ax4.plot(timestamps, est_gyro_bias[:, 2], "b--", label="Est Z")
        ax4.set_ylabel("Gyro bias (rad / s)")
        ax4.set_title("True vs Estimated Gyro bias")
        ax4.legend()
        ax4.grid(True)

        # Plot positions
        ax5.plot(timestamps, true_accelerometer_bias[:, 0], "r-", label="True X")
        ax5.plot(timestamps, true_accelerometer_bias[:, 1], "g-", label="True Y")
        ax5.plot(timestamps, true_accelerometer_bias[:, 2], "b-", label="True Z")
        ax5.plot(timestamps, est_accelerometer_bias[:, 0], "r--", label="Est X")
        ax5.plot(timestamps, est_accelerometer_bias[:, 1], "g--", label="Est Y")
        ax5.plot(timestamps, est_accelerometer_bias[:, 2], "b--", label="Est Z")
        ax5.set_ylabel("Accelerometer bias (m / s^2)")
        ax5.set_title("True vs Estimated Accelerometer bias")
        ax5.legend()
        ax5.grid(True)

        plt.tight_layout()
        plt.show()

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 72:  # h
            self.head_control_mode = not self.head_control_mode
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if not self.head_control_mode:
            if keycode == 265:  # arrow up
                lin_vel_x = self.COMMANDS_RANGE_X[1]
            if keycode == 264:  # arrow down
                lin_vel_x = self.COMMANDS_RANGE_X[0]
            if keycode == 263:  # arrow left
                lin_vel_y = self.COMMANDS_RANGE_Y[1]
            if keycode == 262:  # arrow right
                lin_vel_y = self.COMMANDS_RANGE_Y[0]
            if keycode == 81:  # a
                ang_vel = self.COMMANDS_RANGE_THETA[1]
            if keycode == 69:  # e
                ang_vel = self.COMMANDS_RANGE_THETA[0]
            if keycode == 80:  # p
                self.phase_frequency_factor += 0.1
            if keycode == 59:  # m
                self.phase_frequency_factor -= 0.1
        else:
            neck_pitch = 0
            head_pitch = 0
            head_yaw = 0
            head_roll = 0
            if keycode == 265:  # arrow up
                head_pitch = self.NECK_PITCH_RANGE[1]
            if keycode == 264:  # arrow down
                head_pitch = self.NECK_PITCH_RANGE[0]
            if keycode == 263:  # arrow left
                head_yaw = self.HEAD_YAW_RANGE[1]
            if keycode == 262:  # arrow right
                head_yaw = self.HEAD_YAW_RANGE[0]
            if keycode == 81:  # a
                head_roll = self.HEAD_ROLL_RANGE[1]
            if keycode == 69:  # e
                head_roll = self.HEAD_ROLL_RANGE[0]

            self.commands[3] = neck_pitch
            self.commands[4] = head_pitch
            self.commands[5] = head_yaw
            self.commands[6] = head_roll

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        if not self.standing:
                            self.imitation_i += 1.0 * self.phase_frequency_factor
                            self.imitation_i = (
                                self.imitation_i % self.PRM.nb_steps_in_period
                            )
                            # print(self.PRM.nb_steps_in_period)
                            # exit()
                            self.imitation_phase = np.array(
                                [
                                    np.cos(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                    np.sin(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                ]
                            )
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                            correct_kin=True,  # You can run this less often to save compute.
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))
        
        self.plot_comparisons()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument(
        "--urdf_model_path",
        type=str,
        required=True
    )
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing, urdf_model_path = args.urdf_model_path
    )
    mjinfer.run()
