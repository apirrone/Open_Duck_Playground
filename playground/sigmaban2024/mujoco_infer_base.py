import mujoco
import numpy as np
from etils import epath
from playground.sigmaban2024 import base


class MJInferBase:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_string(
            epath.Path(model_path).read_text(), assets=base.get_assets()
        )
        print(model_path)

        self.sim_dt = 0.002
        self.decimation = 10
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.num_dofs = self.model.nu
        self.floating_base_name = [
            self.model.jnt(k).name
            for k in range(0, self.model.njnt)
            if self.model.jnt(k).type == 0
        ][0]  # assuming only one floating object!
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

        self._floating_base_id = self.model.joint(self.floating_base_name).id

        # self.all_joint_no_backlash_ids=np.zeros(7+self.model.nu)
        all_idx = self.backlash_joint_ids + list(
            range(self._floating_base_qpos_addr, self._floating_base_qpos_addr + 7)
        )
        all_idx.sort()

        # self.all_joint_no_backlash_ids=[idx for idx in self.all_joint_ids if idx not in self.backlash_joint_ids]+list(range(self._floating_base_add,self._floating_base_add+7))
        self.all_joint_no_backlash_ids = [idx for idx in all_idx]

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

        self.gravity_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "upvector"
        )
        self.gravity_dimensions = 3

        self.init_pos = np.array(
            self.get_all_joints_qpos(self.model.keyframe("home").qpos)
        )  # pose of all the joints (no floating base)
        self.default_actuator = self.model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)
        self.motor_targets = self.default_actuator
        self.prev_motor_targets = self.default_actuator

        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.default_actuator

    def get_body_id_from_name(self, name: str) -> int:
        """Return the id of a specified body"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

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

    # def get_gravity(self, data):
    #     return data.site_xmat[self.imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])

    def get_gravity(self, data):
        return data.sensordata[
            self.gravity_id : self.gravity_id + self.gravity_dimensions
        ]

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
        left_contact = self.check_contact(
            data, "left_foot___list_t0v6opd9rekumc_default", "floor"
        )
        right_contact = self.check_contact(data, "right_foot_", "floor")
        return left_contact, right_contact
