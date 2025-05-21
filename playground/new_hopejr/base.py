# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base classes for HopeJR. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from . import constants


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, constants.ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, constants.ROOT_PATH / "xmls" / "assets")
    path = constants.ROOT_PATH
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


class HopeJREnv(mjx_env.MjxEnv):
    """Base class for HopeJR environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        print(f"xml: {xml_path}")
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt

        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path
        self.floating_base_name= [self._mj_model.jnt(k).name for k in range(0, self._mj_model.njnt) if self._mj_model.jnt(k).type == 0][0] #assuming only one floating object!
        self.actuator_names = [
            self._mj_model.actuator(k).name for k in range(0, self._mj_model.nu)
        ]  # will be useful to get only the actuators we care about
        self.joint_names = [ #njnt = all joints (including floating base, actuators and backlash joints)
            self._mj_model.jnt(k).name for k in range(0, self._mj_model.njnt)
        ]  # all the joint (including the backlash joints)
        self.backlash_joint_names = [
            j for j in self.joint_names if j not in self.actuator_names and j not in self.floating_base_name
        ]  # only the dummy backlash joint
        self.all_joint_ids = [self.get_joint_id_from_name(n) for n in self.joint_names]
        self.all_joint_qpos_addr = [self.get_joint_addr_from_name(n) for n in self.joint_names]

        self.actuator_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = [
            self.get_joint_addr_from_name(n) for n in self.actuator_names
        ]

        self.backlash_joint_ids=[
            self.get_joint_id_from_name(n) for n in self.backlash_joint_names
        ]

        self.backlash_joint_qpos_addr=[
            self.get_joint_addr_from_name(n) for n in self.backlash_joint_names
        ]

        self.all_qvel_addr=jp.array([self._mj_model.jnt_dofadr[jad] for jad in self.all_joint_ids])
        self.actuator_qvel_addr=jp.array([self._mj_model.jnt_dofadr[jad] for jad in self.actuator_joint_ids])

        self.actuator_joint_dict = {
            n: self.get_joint_id_from_name(n) for n in self.actuator_names
        }

        self._floating_base_qpos_addr = self._mj_model.jnt_qposadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][
            0
        ]  # Assuming there is only one floating base! the jnt_type==0 is a floating joint. 3 is a hinge

        self._floating_base_qvel_addr = self._mj_model.jnt_dofadr[
            jp.where(self._mj_model.jnt_type == 0)
        ][
            0
        ]  # Assuming there is only one floating base! the jnt_type==0 is a floating joint. 3 is a hinge

        self._floating_base_id = self._mj_model.joint(self.floating_base_name).id

        # self.all_joint_no_backlash_ids=jp.zeros(7+self._mj_model.nu)
        # all_idx=self.actuator_joint_ids+list(range(self._floating_base_qpos_addr,self._floating_base_qpos_addr+7))
        # all_idx=jp.array(all_idx).sort()
        all_idx=self.actuator_joint_ids+list([self.get_joint_id_from_name("trunk_assembly_freejoint")])
        all_idx=jp.array(all_idx).sort()
        # self.all_joint_no_backlash_ids=[idx for idx in self.all_joint_ids if idx not in self.backlash_joint_ids]+list(range(self._floating_base_add,self._floating_base_add+7))
        self.all_joint_no_backlash_ids=[idx for idx in all_idx]
        # print(f"ALL: {self.all_joint_no_backlash_ids} back_id: {self.backlash_joint_ids} base_id: {list(range(self._floating_base_qpos_addr,self._floating_base_qpos_addr+7))}")

        self.backlash_idx_to_add = []

        for i, actuator_name in enumerate(self.actuator_names):
            if actuator_name + "_backlash" not in self.backlash_joint_names:
                self.backlash_idx_to_add.append(i)

        print(f"actuators: {self.actuator_names}")
        print(f"joints: {self.joint_names}")
        print(f"backlash joints: {self.backlash_joint_names}")
        print(f"actuator joints ids: {self.actuator_joint_ids}")
        print(f"actuator joints dict: {self.actuator_joint_dict}")
        print(f"floating qpos addr: {self._floating_base_qpos_addr} qvel addr: {self._floating_base_qvel_addr}")



    def get_actuator_id_from_name(self, name: str) -> int:
        """Return the id of a specified actuator"""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint_id_from_name(self, name: str) -> int:
        """Return the id of a specified joint"""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)


    def get_joint_addr_from_name(self, name: str) -> int:
        """Return the address of a specified joint"""
        return self._mj_model.joint(name).qposadr

    def get_dof_id_from_name(self, name: str) -> int:
        """Return the id of a specified dof"""
        return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_DOF, name)


    def get_actuator_joint_qpos_from_name(self, data: jax.Array, name: str) -> jax.Array:
        """Return the qpos of a given actual joint"""
        addr = self._mj_model.jnt_qposadr[self.actuator_joint_dict[name]]
        return data[addr]

    def get_actuator_joints_qpos_addr(self) -> jax.Array:
        """Return the all the idx of actual joints"""
        addr = jp.array(
            [self._mj_model.jnt_qposadr[idx] for idx in self.actuator_joint_ids]
        )
        return addr

    def get_floating_base_qpos(self, data:jax.Array) -> jax.Array:
        return data[self._floating_base_qpos_addr:self._floating_base_qvel_addr+7]

    def get_floating_base_qvel(self, data:jax.Array) -> jax.Array:
        return data[self._floating_base_qvel_addr:self._floating_base_qvel_addr+6]


    def set_floating_base_qpos(self, new_qpos:jax.Array, qpos:jax.Array) -> jax.Array:
        return qpos.at[self._floating_base_qpos_addr:self._floating_base_qpos_addr+7].set(new_qpos)

    def set_floating_base_qvel(self, new_qvel:jax.Array, qvel:jax.Array) -> jax.Array:
        return qvel.at[self._floating_base_qvel_addr:self._floating_base_qvel_addr+6].set(new_qvel)


    def exclude_backlash_joints_addr(self) -> jax.Array:
        """Return the all the idx of actual joints and floating base"""
        addr = jp.array(
            [self._mj_model.jnt_qposadr[idx] for idx in self.all_joint_no_backlash_ids]
        )
        return addr


    def get_all_joints_addr(self) -> jax.Array:
        """Return the all the idx of all joints"""
        addr = jp.array([self._mj_model.jnt_qposadr[idx] for idx in self.all_joint_ids])
        return addr

    def get_actuator_joints_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of actual joints"""
        return data[self.get_actuator_joints_qpos_addr()]

    def set_actuator_joints_qpos(self, new_qpos: jax.Array, qpos: jax.Array) -> jax.Array:
        """Set the qpos only for the actual joints (omit the backlash joint)"""
        return qpos.at[self.get_actuator_joints_qpos_addr()].set(new_qpos)

    def get_actuator_backlash_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of backlash joints"""
        if self.backlash_joint_qpos_addr == []:
            return jp.array([])
        return data[jp.array(self.backlash_joint_qpos_addr)]


    def get_actuator_joints_qvel(self, data: jax.Array) -> jax.Array:
        """Return the all the qvel of actual joints"""
        return data[self.actuator_qvel_addr]

    def set_actuator_joints_qvel(self, new_qvel: jax.Array, qvel: jax.Array) -> jax.Array:
        """Set the qvel only for the actual joints (omit the backlash joint)"""
        return qvel.at[self.actuator_qvel_addr].set(new_qvel)

    def get_all_joints_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of all joints"""
        return data[self.get_all_joints_addr()]

    def get_all_joints_qvel(self, data: jax.Array) -> jax.Array:
        """Return the all the qvel of all joints"""
        return data[self.all_qvel_addr]

    def get_joints_nobacklash_qpos(self, data: jax.Array) -> jax.Array:
        """Return the all the qpos of actual joints with the floating base"""
        return data[self.exclude_backlash_joints_addr()]

    def set_complete_qpos_from_joints(self, no_backlash_qpos: jax.Array, full_qpos: jax.Array) -> jax.Array:
        """In the case of backlash joints, we want to ignore them (remove them) but we still need to set the complete state incuding them"""
        full_qpos.at[self.exclude_backlash_joints_addr()].set(no_backlash_qpos)
        return jp.array(full_qpos)

    # Sensor readings.
    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Return the gravity vector in the world frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, constants.GRAVITY_SENSOR)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        """Return the angular velocity of the robot in the world frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        """Return the linear velocity of the robot in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        """Return the accelerometer readings in the local frame."""
        return mjx_env.get_sensor_data(
            self.mj_model, data, constants.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        """Return the gyroscope readings in the local frame."""
        return mjx_env.get_sensor_data(self.mj_model, data, constants.GYRO_SENSOR)

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        """Return the position of the feet in the world frame."""
        return jp.vstack(
            [
                mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
                for sensor_name in constants.FEET_POS_SENSOR
            ]
        )

    # Accessors.

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
