import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from etils import epath
from scipy.spatial.transform import Rotation as sp_R
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

# from playground.open_duck_mini_v2 import constants
from playground.open_duck_mini_v2 import base

from playground.common.contact_aided_invariant_ekf.contact_aided_invariant_ekf import (
    RightInEKF,
)
from playground.common.contact_aided_invariant_ekf.robot_state import RobotState
from playground.common.contact_aided_invariant_ekf.noise_params import NoiseParams
from playground.common.contact_aided_invariant_ekf.drake_kinematics import DrakeKinematics

USE_MOTOR_SPEED_LIMITS = True


class MjInfer:
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool, urdf_model_path: str
    ):
        self.model = mujoco.MjModel.from_xml_string(
            epath.Path(model_path).read_text(), assets=base.get_assets()
        )
        print(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.obs_history_length = 3

        self.obs_history = np.zeros(self.obs_history_length*45)

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        NUM_DOFS = self.model.nu
        self.floating_base_name = [
            self.model.jnt(k).name
            for k in range(0, self.model.njnt)
            if self.model.jnt(k).type == 0
        ][
            0
        ]  # assuming only one floating object!
        self.actuator_names = [
            self.model.actuator(k).name for k in range(0, self.model.nu)
        ]  # will be useful to get only the actuators we care about
        self.joint_names = [  # njnt = all joints (including floating base, actuators and backlash joints)
            self.model.jnt(k).name for k in range(0, self.model.njnt)
        ]  # all the joint (including the backlash joints)
        self.backlash_joint_names = [
            j
            for j in self.joint_names
            if j not in self.actuator_names and j not in self.floating_base_name
        ]  # only the dummy backlash joint
        self.all_joint_ids = [self.get_joint_id_from_name(n) for n in self.joint_names]
        self.all_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.joint_names
        ]

        self.actuator_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.actuator_names
        ]

        self.backlash_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.backlash_joint_names
        ]

        self.backlash_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.backlash_joint_names
        ]

        self.all_qvel_addr = np.array(
            [self.model.jnt_dofadr[jad] for jad in self.all_joint_ids]
        )
        self.actuator_qvel_addr = np.array(
            [self.model.jnt_dofadr[jad] for jad in self.actuator_joint_ids]
        )

        self.actuator_joint_dict = {
            n: self.get_joint_id_from_name(n) for n in self.actuator_names
        }

        self._floating_base_qpos_addr = self.model.jnt_qposadr[
            np.where(self.model.jnt_type == 0)
        ][
            0
        ]  # Assuming there is only one floating base! the jnt_type==0 is a floating joint. 3 is a hinge

        self._floating_base_qvel_addr = self.model.jnt_dofadr[
            np.where(self.model.jnt_type == 0)
        ][
            0
        ]  # Assuming there is only one floating base! the jnt_type==0 is a floating joint. 3 is a hinge

        self.actuator_name_to_idx_map = {
            n: self.get_joint_addr_from_name(n) for n in self.actuator_names
        }

        self._floating_base_id = self.model.joint(self.floating_base_name).id

        # self.all_joint_no_backlash_ids=np.zeros(7+self.model.nu)
        all_idx = self.backlash_joint_ids + list(
            range(self._floating_base_qpos_addr, self._floating_base_qpos_addr + 7)
        )
        all_idx.sort()

        # self.all_joint_no_backlash_ids=[idx for idx in self.all_joint_ids if idx not in self.backlash_joint_ids]+list(range(self._floating_base_add,self._floating_base_add+7))
        self.all_joint_no_backlash_ids = [idx for idx in all_idx]

        self.sim_dt = 0.002
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.15, 0.2]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(NUM_DOFS)
        self.last_last_action = np.zeros(NUM_DOFS)
        self.last_last_last_action = np.zeros(NUM_DOFS)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.decimation = 10

        self.init_pos = np.array(
            self.get_all_joints_qpos(self.model.keyframe("home").qpos)
        )  # pose of all the joints (no floating base)
        self.default_actuator = self.model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)
        self.motor_targets = self.default_actuator
        self.prev_motor_targets = self.default_actuator

        # orientation
        # data.qpos[3 : 3 + 4] = [1, 0, 0.0, 0]
        # data.qpos[7:] = init_pos
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.default_actuator

        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.gyro_addr = self.model.sensor_adr[self.gyro_id]
        self.gyro_dimensions = 3

        self.accelerometer_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer"
        )
        self.accelerometer_dimensions = 3
        self.accelerometer_addr = self.model.sensor_adr[self.accelerometer_id]

        self.linvel_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel"
        )
        self.linvel_dimensions = 3

        self.imu_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0
        self.phase_active = True
        self.waiting_for_zero = False

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

        # Gather body and feet id
        self.imu_id = self.data.body("imu").id
        self.feet_ids = {
            0: self.data.body("foot_assembly").id,
            1: self.data.body("foot_assembly_2").id,
        }

        # If unknown, set to identity.
        initial_pose = self.get_pose_from_id(self.data, self.imu_id)

        # Convert to rotation matrix
        body_rotation_matrix = initial_pose[0:3, 0:3]
        self.ekf_noise_params = NoiseParams() # TODO: put in real
        # self.ekf_noise_params.set_gyroscope_bias_noise(1e-6)
        # ekf_noise_params.set_contact_noise(0.5)
        ekf_robot_state = RobotState()
        ekf_robot_state.set_position(initial_pose[0:3, 3])
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
        self.ekf_imu_measurement_prev = np.zeros(6)
    
    def get_pose_from_id(self, data, id: str) -> np.ndarray:
        pos = data.xpos[id]
        quat = data.xquat[id]
        pose = np.eye(4)
        pose[:3, :3] = sp_R.from_quat(quat, scalar_first=True).as_matrix()
        pose[:3, 3] = pos
        return pose

    def get_actuator_id_from_name(self, name: str) -> int:
        """Return the id of a specified actuator"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint_id_from_name(self, name: str) -> int:
        """Return the id of a specified joint"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def get_joint_addr_from_name(self, name: str) -> int:
        """Return the address of a specified joint"""
        return self.model.joint(name).qposadr

    def get_dof_id_from_name(self, name: str) -> int:
        """Return the id of a specified dof"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_DOF, name)

    def get_actuator_joint_qpos_from_name(
        self, data: np.ndarray, name: str
    ) -> np.ndarray:
        """Return the qpos of a given actual joint"""
        addr = self.model.jnt_qposadr[self.actuator_joint_dict[name]]
        return data[addr]

    def get_actuator_joints_addr(self) -> np.ndarray:
        """Return the all the idx of actual joints"""
        addr = np.array(
            [self.model.jnt_qposadr[idx] for idx in self.actuator_joint_ids]
        )
        return addr

    def get_floating_base_qpos(self, data: np.ndarray) -> np.ndarray:
        return data[self._floating_base_qpos_addr : self._floating_base_qvel_addr + 7]

    def get_floating_base_qvel(self, data: np.ndarray) -> np.ndarray:
        return data[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6]

    def set_floating_base_qpos(
        self, new_qpos: np.ndarray, qpos: np.ndarray
    ) -> np.ndarray:
        qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 7] = (
            new_qpos
        )
        return qpos

    def set_floating_base_qvel(
        self, new_qvel: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6] = (
            new_qvel
        )
        return qvel

    def exclude_backlash_joints_addr(self) -> np.ndarray:
        """Return the all the idx of actual joints and floating base"""
        addr = np.array(
            [self.model.jnt_qposadr[idx] for idx in self.all_joint_no_backlash_ids]
        )
        return addr

    def get_all_joints_addr(self) -> np.ndarray:
        """Return the all the idx of all joints"""
        addr = np.array([self.model.jnt_qposadr[idx] for idx in self.all_joint_ids])
        return addr

    def get_actuator_joints_qpos(self, data: np.ndarray) -> np.ndarray:
        """Return the all the qpos of actual joints"""
        return data[self.get_actuator_joints_addr()]

    def set_actuator_joints_qpos(
        self, new_qpos: np.ndarray, qpos: np.ndarray
    ) -> np.ndarray:
        """Set the qpos only for the actual joints (omit the backlash joint)"""
        qpos[self.get_actuator_joints_addr()] = new_qpos
        return qpos

    def get_actuator_joints_qvel(self, data: np.ndarray) -> np.ndarray:
        """Return the all the qvel of actual joints"""
        return data[self.actuator_qvel_addr]

    def set_actuator_joints_qvel(
        self, new_qvel: np.ndarray, qvel: np.ndarray
    ) -> np.ndarray:
        """Set the qvel only for the actual joints (omit the backlash joint)"""
        qvel[self.actuator_qvel_addr] = new_qvel
        return qvel

    def get_all_joints_qpos(self, data: np.ndarray) -> np.ndarray:
        """Return the all the qpos of all joints"""
        return data[self.get_all_joints_addr()]

    def get_all_joints_qvel(self, data: np.ndarray) -> np.ndarray:
        """Return the all the qvel of all joints"""
        return data[self.all_qvel_addr]

    def get_joints_nobacklash_qpos(self, data: np.ndarray) -> np.ndarray:
        """Return the all the qpos of actual joints with the floating base"""
        return data[self.exclude_backlash_joints_addr()]

    def set_complete_qpos_from_joints(
        self, no_backlash_qpos: np.ndarray, full_qpos: np.ndarray
    ) -> np.ndarray:
        """In the case of backlash joints, we want to ignore them (remove them) but we still need to set the complete state incuding them"""
        full_qpos[self.exclude_backlash_joints_addr()] = no_backlash_qpos
        return np.array(full_qpos)

    def get_sensor(self, data, name, dimensions):
        i = self.model.sensor_name2id(name)
        return data.sensordata[i : i + dimensions]

    def get_gyro(self, data):
        return data.sensordata[self.gyro_addr : self.gyro_addr + self.gyro_dimensions]

    def get_accelerometer(self, data):
        return data.sensordata[
            self.accelerometer_addr : self.accelerometer_addr
            + self.accelerometer_dimensions
        ]

    def get_linvel(self, data):
        return data.sensordata[self.linvel_id : self.linvel_id + self.linvel_dimensions]

    def get_gravity(self, data):
        return data.site_xmat[self.imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])

    def check_contact(self, data, body1_name, body2_name):
        body1_id = data.body(body1_name).id
        body2_id = data.body(body2_name).id

        for i in range(data.ncon):
            try:
                contact = data.contact[i]
            except Exception as e:
                return False

            if (
                self.model.geom_bodyid[contact.geom1] == body1_id
                and self.model.geom_bodyid[contact.geom2] == body2_id
            ) or (
                self.model.geom_bodyid[contact.geom1] == body2_id
                and self.model.geom_bodyid[contact.geom2] == body1_id
            ):
                return True

        return False

    def get_feet_contacts(self, data):
        left_contact = self.check_contact(data, "foot_assembly", "floor")
        right_contact = self.check_contact(data, "foot_assembly_2", "floor")
        return left_contact, right_contact

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
        use_ekf: bool = False,
        correct_kin: bool = False
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        gravity = self.get_gravity(data)
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        ekf_imu_measurement = np.zeros(6)
        ekf_imu_measurement[0:3] = gyro
        ekf_imu_measurement[3:6] = accelerometer

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

        if not use_ekf:
            imu_pose_world = self.get_pose_from_id(data, self.imu_id)
            rotation_world = imu_pose_world[0:3, 0:3]
            translation_world = imu_pose_world[0:3, 3]
            floating_base_vel_world = data.qvel[self._floating_base_qvel_addr:self._floating_base_qvel_addr + 6]
        else:
            rotation_world = estimated_state.get_rotation()
            translation_world = estimated_state.get_position()
            floating_base_vel_world = estimated_state.get_velocity()


        rotation_world_6d = rotation_world[:,0:2].flatten()
        floating_base_vel_body = rotation_world.T @ floating_base_vel_world[0:3]

        obs = np.concatenate(
            [   
                rotation_world_6d,
                # translation_world,
                floating_base_vel_body,
                gyro,
                accelerometer,
                # gravity,
                command,
                joint_angles - self.default_actuator,
                # joint_vel * self.dof_vel_scale,
                self.last_action,
                # self.last_last_action,
                # self.last_last_last_action,
                # self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase,
            ]
        )

        return obs

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
                if self.phase_active:
                    self.phase_active = False
                    self.waiting_for_zero = True
                else:
                    self.phase_active = True
                    self.waiting_for_zero = False
                print(f"Phase advancement {'enabled' if self.phase_active else 'disabled'}")

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
                            # Update phase only if active or waiting to reach zero
                            if self.phase_active or self.waiting_for_zero:
                                self.imitation_i += 1.0 * self.phase_frequency_factor
                                self.imitation_i = (
                                    self.imitation_i % self.PRM.nb_steps_in_period
                                )
                                
                                # If we're waiting for zero and we've reached it, stop phase advancement
                                if self.waiting_for_zero and self.imitation_i < 1.0:
                                    self.imitation_i = 0
                                    self.waiting_for_zero = False
                                    print("Phase advancement stopped at zero")
                                    
                            self.imitation_phase = np.array(
                                [
                                    np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                                    np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                                ]
                            )
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                            use_ekf=True,
                            correct_kin=True
                        )
                        magic_number = 45 + 6 + 3 # 6d pose, lin vel
                        self.obs_history = np.roll(self.obs_history, magic_number)
                        self.obs_history[:magic_number] = obs
                        obs = self.obs_history

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
