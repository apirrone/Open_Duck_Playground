"""
Set of commonly used rewards
For examples on how to use some rewards, look at https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/berkeley_humanoid/joystick.py
"""

import jax
import jax.numpy as jp


# Tracking rewards.
def reward_tracking_lin_vel(
    commands: jax.Array,
    local_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    # lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    # return jp.nan_to_num(jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma))
    y_tol = 0.1
    error_x = jp.square(commands[0] - local_vel[0])
    error_y = jp.clip(jp.abs(local_vel[1] - commands[1]) - y_tol, 0.0, None)
    lin_vel_error = error_x + jp.square(error_y)
    return jp.nan_to_num(jp.exp(-lin_vel_error / tracking_sigma))


def reward_tracking_ang_vel(
    commands: jax.Array,
    ang_vel: jax.Array,
    tracking_sigma: float,
) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.nan_to_num(jp.exp(-ang_vel_error / tracking_sigma))


# Base-related rewards.


def cost_lin_vel_z(global_linvel) -> jax.Array:
    return jp.nan_to_num(jp.square(global_linvel[2]))


def cost_ang_vel_xy(global_angvel) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(global_angvel[:2])))


def cost_orientation(torso_zaxis: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torso_zaxis[:2])))


def cost_base_height(base_height: jax.Array, base_height_target: float) -> jax.Array:
    return jp.nan_to_num(jp.square(base_height - base_height_target))


def reward_base_y_swing(
    base_y_speed: jax.Array,
    freq: float,
    amplitude: float,
    t: float,
    tracking_sigma: float,
) -> jax.Array:
    target_y_speed = amplitude * jp.sin(2 * jp.pi * freq * t)
    y_speed_error = jp.square(target_y_speed - base_y_speed)
    return jp.nan_to_num(jp.exp(-y_speed_error / tracking_sigma))


# Energy related rewards.


def cost_torques(torques: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torques)))
    # return jp.nan_to_num(jp.sum(jp.abs(torques)))


def cost_energy(qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)))


def cost_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    c1 = jp.nan_to_num(jp.sum(jp.square(act - last_act)))
    return c1


# Other rewards.


def cost_joint_pos_limits(
    qpos: jax.Array, soft_lowers: float, soft_uppers: float
) -> jax.Array:
    out_of_limits = -jp.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - soft_uppers, 0.0, None)
    return jp.nan_to_num(jp.sum(out_of_limits))


def cost_stand_still(
    commands: jax.Array,
    qpos: jax.Array,
    qvel: jax.Array,
    default_pose: jax.Array,
) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands[:3])
    pose_cost = jp.sum(jp.abs(qpos[5:9] - default_pose[5:9]))  # ignore head
    vel_cost = jp.sum(jp.abs(qvel[5:9]))  # ignore head
    return jp.nan_to_num(pose_cost + vel_cost) * (cmd_norm < 0.01)


def cost_termination(done: jax.Array) -> jax.Array:
    return done


# def quaternion_angle(q1, q2):
#     #  according to chatgpt :)
#     dot_product = jp.dot(q1, q2)
#     dot_product = jp.clip(dot_product, -1.0, 1.0)
#     angle = 2 * jp.arccos(jp.abs(dot_product))
#     return angle


def reward_imitation(
    base_qvel: jax.Array,
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    use_imitation_reward: bool = False,
) -> jax.Array:
    if not use_imitation_reward:
        return jp.nan_to_num(0.0)

    # TODO don't reward for moving when the command is zero.
    cmd_norm = jp.linalg.norm(cmd[:3])

    w_torso_pos = 1.0
    w_torso_orientation = 1.0
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 1.0
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_joint_pos = 15.0
    w_joint_vel = 1.0e-3
    w_contact = 1.0

    #  TODO : double check if the slices are correct
    linear_vel_slice_start = 34
    linear_vel_slice_end = 37

    angular_vel_slice_start = 37
    angular_vel_slice_end = 40

    joint_pos_slice_start = 0
    joint_pos_slice_end = 16

    joint_vels_slice_start = 16
    joint_vels_slice_end = 32

    # root_pos_slice_start = 0
    # root_pos_slice_end = 3

    # root_quat_slice_start = 3
    # root_quat_slice_end = 7

    # left_toe_pos_slice_start = 23
    # left_toe_pos_slice_end = 26

    # right_toe_pos_slice_start = 26
    # right_toe_pos_slice_end = 29

    foot_contacts_slice_start = 32
    foot_contacts_slice_end = 34

    # ref_base_pos = reference_frame[root_pos_slice_start:root_pos_slice_end]
    # base_pos = qpos[:3]

    # ref_base_orientation_quat = reference_frame[root_quat_slice_start:root_quat_slice_end]
    # ref_base_orientation_quat = ref_base_orientation_quat / jp.linalg.norm(ref_base_orientation_quat)  # normalize the quat
    # base_orientation = qpos[3:7]
    # base_orientation = base_orientation / jp.linalg.norm(base_orientation)  # normalize the quat

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]

    # remove the neck and head
    ref_joint_pos = jp.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    joint_pos = jp.concatenate([joints_qpos[:5], joints_qpos[9:]])

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    # remove the neck and head
    ref_joint_vels = jp.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    joint_vel = jp.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # ref_left_toe_pos = reference_frame[left_toe_pos_slice_start:left_toe_pos_slice_end]
    # ref_right_toe_pos = reference_frame[right_toe_pos_slice_start:right_toe_pos_slice_end]

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]

    # reward
    # torso_pos_rew = jp.exp(-200.0 * jp.sum(jp.square(base_pos[:2] - ref_base_pos[:2]))) * w_torso_pos

    # real quaternion angle doesn't have the expected  effect, switching back for now
    # torso_orientation_rew = jp.exp(-20 * self.quaternion_angle(base_orientation, ref_base_orientation_quat)) * w_torso_orientation
    # torso_orientation_rew = jp.exp(-20.0 * jp.sum(jp.square(base_orientation - ref_base_orientation_quat))) * w_torso_orientation

    lin_vel_xy_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[:2] - ref_base_lin_vel[:2])))
        * w_lin_vel_xy
    )
    lin_vel_z_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[2] - ref_base_lin_vel[2])))
        * w_lin_vel_z
    )

    ang_vel_xy_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[:2] - ref_base_ang_vel[:2])))
        * w_ang_vel_xy
    )
    ang_vel_z_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[2] - ref_base_ang_vel[2])))
        * w_ang_vel_z
    )

    joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) * w_joint_pos
    joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) * w_joint_vel

    ref_foot_contacts = jp.where(
        ref_foot_contacts > 0.5,
        jp.ones_like(ref_foot_contacts),
        jp.zeros_like(ref_foot_contacts),
    )
    contact_rew = jp.sum(contacts == ref_foot_contacts) * w_contact

    # reward = torso_pos_rew + torso_orientation_rew +  lin_vel_xy_rew + lin_vel_z_rew + ang_vel_xy_rew + ang_vel_z_rew + joint_pos_rew + joint_vel_rew + contact_rew
    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
    )
    # reward = joint_pos_rew + joint_vel_rew + contact_rew # trying without the lin and ang vel because they can compete with the tracking rewards
    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)


def reward_alive() -> jax.Array:
    return jp.array(1.0)


# Pose-related rewards.


def cost_head_pos(
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    cmd: jax.Array,
) -> jax.Array:

    head_cmd = cmd[3:]
    head_pos = joints_qpos[5:9]
    head_vel = joints_qvel[5:9]

    target_head_qvel = jp.zeros_like(head_cmd)

    head_pos_error = jp.sum(jp.square(head_pos - head_cmd))

    head_vel_error = jp.sum(jp.square(head_vel - target_head_qvel))

    return jp.nan_to_num(head_pos_error + head_vel_error)


# FIXME
def cost_joint_deviation_hip(
    qpos: jax.Array, cmd: jax.Array, hip_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    cost = jp.sum(jp.abs(qpos[hip_indices] - default_pose[hip_indices]))
    cost *= jp.abs(cmd[1]) > 0.1
    return jp.nan_to_num(cost)


# FIXME
def cost_joint_deviation_knee(
    qpos: jax.Array, knee_indices: jax.Array, default_pose: jax.Array
) -> jax.Array:
    return jp.nan_to_num(
        jp.sum(jp.abs(qpos[knee_indices] - default_pose[knee_indices]))
    )


# FIXME
def cost_pose(
    qpos: jax.Array, default_pose: jax.Array, weights: jax.Array
) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(qpos - default_pose) * weights))


# Feet related rewards.


# FIXME
def cost_feet_slip(contact: jax.Array, global_linvel: jax.Array) -> jax.Array:
    body_vel = global_linvel[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return jp.nan_to_num(reward)


# FIXME
def cost_feet_clearance(
    feet_vel: jax.Array,
    foot_pos: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    # feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - max_foot_height)
    return jp.nan_to_num(jp.sum(delta * vel_norm))


# FIXME
def cost_feet_height(
    swing_peak: jax.Array,
    first_contact: jax.Array,
    max_foot_height: float,
) -> jax.Array:
    error = swing_peak / max_foot_height - 1.0
    return jp.nan_to_num(jp.sum(jp.square(error) * first_contact))


# FIXME
def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    commands: jax.Array,
    threshold_min: float = 0.1,  # 0.2
    threshold_max: float = 0.5,
) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands[:3])
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)


# FIXME
def reward_feet_phase(
    foot_pos: jax.Array,
    rz: jax.Array,
) -> jax.Array:
    # Reward for tracking the desired foot height.
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    # rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return jp.nan_to_num(reward)
