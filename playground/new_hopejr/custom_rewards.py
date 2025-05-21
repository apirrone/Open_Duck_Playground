import jax
import jax.numpy as jp


def reward_imitation(
    base_qpos: jax.Array,
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

    # Mansin' weights
    # w_torso_pos = 0.0
    # w_torso_orientation = 0.0
    # w_lin_vel_xy = 0.0
    # w_lin_vel_z = 0.0
    # w_ang_vel_xy = 0.0
    # w_ang_vel_z = 0.0
    # w_joint_pos = 15.0
    # w_joint_vel = 1.0e-3
    # w_contact = 5.0


    # dimensions_names = [
    #     0  "pos head_yaw",
    #     1  "pos head_pitch",
    #     2  "pos left_shoulder_pitch",
    #     3  "pos left_shoulder_roll",
    #     4  "pos left_elbow",
    #     5  "pos right_shoulder_pitch",
    #     6  "pos right_shoulder_roll",
    #     7  "pos right_elbow",
    #     8  "pos left_hip_yaw",
    #     9  "pos left_hip_roll",
    #     10 "pos left_hip_pitch",
    #     11 "pos left_knee",
    #     12 "pos left_ankle_pitch",
    #     13 "pos left_ankle_roll",
    #     14 "pos right_hip_yaw",
    #     15 "pos right_hip_roll",
    #     16 "pos right_hip_pitch",
    #     17 "pos right_knee",
    #     18 "pos right_ankle_pitch",
    #     19 "pos right_ankle_roll",

    #     20 "vel head_yaw",
    #     21 "vel head_pitch",
    #     22 "vel left_shoulder_pitch",
    #     23 "vel left_shoulder_roll",
    #     24 "vel left_elbow",
    #     25 "vel right_shoulder_pitch",
    #     26 "vel right_shoulder_roll",
    #     27 "vel right_elbow",
    #     28 "vel left_hip_yaw",
    #     29 "vel left_hip_roll",
    #     30 "vel left_hip_pitch",
    #     31 "vel left_knee",
    #     32 "vel left_ankle_pitch",
    #     33 "vel left_ankle_roll",
    #     34 "vel right_hip_yaw",
    #     35 "vel right_hip_roll",
    #     36 "vel right_hip_pitch",
    #     37 "vel right_knee",
    #     38 "vel right_ankle_pitch",
    #     39 "vel right_ankle_roll",

    #     40 "foot_contacts left",
    #     41 "foot_contacts right",

    #     42 "base_linear_vel x",
    #     43 "base_linear_vel y",
    #     44 "base_linear_vel z",

    #     45 "base_angular_vel x",
    #     46 "base_angular_vel y",
    #     47 "base_angular_vel z",
    # ]

    #  TODO : double check if the slices are correct
    joint_pos_slice_start = 0
    joint_pos_slice_end = 27

    joint_vels_slice_start = 27
    joint_vels_slice_end = 54

    foot_contacts_slice_start = 54
    foot_contacts_slice_end = 56

    linear_vel_slice_start = 56
    linear_vel_slice_end = 59

    angular_vel_slice_start = 59
    angular_vel_slice_end = 62

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    joint_pos = joints_qpos

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    joint_vel = joints_qvel

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]

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

    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
        # + torso_orientation_rew
    )

    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)


def cost_feet_rectangle_contact(feet_rectangle_contact:jax.Array):
    return jp.nan_to_num(-jp.sum(feet_rectangle_contact))
