import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", type=str, required=False, default="mujoco_saved_obs.pkl"
)
args = parser.parse_args()


init_pos = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -0.48,
        1.0,
        -0.48,
        0,
        0,
        0,
        -0.48,
        1.0,
        -0.48,
        0,
    ]
)

joints_order = [
    "head_yaw",
    "head_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
]

obses = pickle.load(open(args.data, "rb"))

num_dofs = len(joints_order)
dof_poses = []  # (dof, num_obs)
actions = []  # (dof, num_obs)

for i in range(num_dofs):
    print(i)
    dof_poses.append([])
    actions.append([])
    for obs in obses:
        dof_poses[i].append(obs[13 : 13 + num_dofs][i])
        actions[i].append(obs[26 : 26 + num_dofs][i])

# plot action vs dof pos

nb_dofs = len(dof_poses)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(actions[i * nb_cols + j], label="action")
        axs[i, j].plot(dof_poses[i * nb_cols + j], label="dof_pos")
        axs[i, j].legend()
        axs[i, j].set_title(f"{joints_order[i * nb_cols + j]}")

fig.suptitle(f"{args.data}")
plt.show()

obses_names = [
    "gyro x",
    "gyro y",
    "gyro z",
    "gravity x",
    "gravity y",
    "gravity z",
    # commands
    "command 0",
    "command 1",
    "command 2",
    "command 3",
    "command 4",
    "command 5",
    "command 6",
    # dof pos
    "pos_" + str(joints_order[0]),
    "pos_" + str(joints_order[1]),
    "pos_" + str(joints_order[2]),
    "pos_" + str(joints_order[3]),
    "pos_" + str(joints_order[4]),
    "pos_" + str(joints_order[5]),
    "pos_" + str(joints_order[6]),
    "pos_" + str(joints_order[7]),
    "pos_" + str(joints_order[8]),
    "pos_" + str(joints_order[9]),
    "pos_" + str(joints_order[10]),
    "pos_" + str(joints_order[11]),
    "pos_" + str(joints_order[12]),
    "pos_" + str(joints_order[13]),
    "pos_" + str(joints_order[14]),
    "pos_" + str(joints_order[15]),
    "pos_" + str(joints_order[16]),
    "pos_" + str(joints_order[17]),
    "pos_" + str(joints_order[18]),
    "pos_" + str(joints_order[19]),
    # dof vel
    "vel_" + str(joints_order[0]),
    "vel_" + str(joints_order[1]),
    "vel_" + str(joints_order[2]),
    "vel_" + str(joints_order[3]),
    "vel_" + str(joints_order[4]),
    "vel_" + str(joints_order[5]),
    "vel_" + str(joints_order[6]),
    "vel_" + str(joints_order[7]),
    "vel_" + str(joints_order[8]),
    "vel_" + str(joints_order[9]),
    "vel_" + str(joints_order[10]),
    "vel_" + str(joints_order[11]),
    "vel_" + str(joints_order[12]),
    "vel_" + str(joints_order[13]),
    "vel_" + str(joints_order[14]),
    "vel_" + str(joints_order[15]),
    "vel_" + str(joints_order[16]),
    "vel_" + str(joints_order[17]),
    "vel_" + str(joints_order[18]),
    "vel_" + str(joints_order[19]),
    # action
    "last_action_" + str(joints_order[0]),
    "last_action_" + str(joints_order[1]),
    "last_action_" + str(joints_order[2]),
    "last_action_" + str(joints_order[3]),
    "last_action_" + str(joints_order[4]),
    "last_action_" + str(joints_order[5]),
    "last_action_" + str(joints_order[6]),
    "last_action_" + str(joints_order[7]),
    "last_action_" + str(joints_order[8]),
    "last_action_" + str(joints_order[9]),
    "last_action_" + str(joints_order[10]),
    "last_action_" + str(joints_order[11]),
    "last_action_" + str(joints_order[12]),
    "last_action_" + str(joints_order[13]),
    "last_action_" + str(joints_order[14]),
    "last_action_" + str(joints_order[15]),
    "last_action_" + str(joints_order[16]),
    "last_action_" + str(joints_order[17]),
    "last_action_" + str(joints_order[18]),
    "last_action_" + str(joints_order[19]),
    "last_last_action_" + str(joints_order[0]),
    "last_last_action_" + str(joints_order[1]),
    "last_last_action_" + str(joints_order[2]),
    "last_last_action_" + str(joints_order[3]),
    "last_last_action_" + str(joints_order[4]),
    "last_last_action_" + str(joints_order[5]),
    "last_last_action_" + str(joints_order[6]),
    "last_last_action_" + str(joints_order[7]),
    "last_last_action_" + str(joints_order[8]),
    "last_last_action_" + str(joints_order[9]),
    "last_last_action_" + str(joints_order[10]),
    "last_last_action_" + str(joints_order[11]),
    "last_last_action_" + str(joints_order[12]),
    "last_last_action_" + str(joints_order[13]),
    "last_last_action_" + str(joints_order[14]),
    "last_last_action_" + str(joints_order[15]),
    "last_last_action_" + str(joints_order[16]),
    "last_last_action_" + str(joints_order[17]),
    "last_last_action_" + str(joints_order[18]),
    "last_last_action_" + str(joints_order[19]),
    "last_last_last_action_" + str(joints_order[0]),
    "last_last_last_action_" + str(joints_order[1]),
    "last_last_last_action_" + str(joints_order[2]),
    "last_last_last_action_" + str(joints_order[3]),
    "last_last_last_action_" + str(joints_order[4]),
    "last_last_last_action_" + str(joints_order[5]),
    "last_last_last_action_" + str(joints_order[6]),
    "last_last_last_action_" + str(joints_order[7]),
    "last_last_last_action_" + str(joints_order[8]),
    "last_last_last_action_" + str(joints_order[9]),
    "last_last_last_action_" + str(joints_order[10]),
    "last_last_last_action_" + str(joints_order[11]),
    "last_last_last_action_" + str(joints_order[12]),
    "last_last_last_action_" + str(joints_order[13]),
    "last_last_last_action_" + str(joints_order[14]),
    "last_last_last_action_" + str(joints_order[15]),
    "last_last_last_action_" + str(joints_order[16]),
    "last_last_last_action_" + str(joints_order[17]),
    "last_last_last_action_" + str(joints_order[18]),
    "last_last_last_action_" + str(joints_order[19]),
    "contact left",
    "contact right",
    "imitation_phase 1",
    "imitation_phase 2",
    # ref (ignored)
]
# print(len(obses_names))
# exit()


# obses = [[56 obs at time 0], [56 obs at time 1], ...]

nb_obs = len(obses[0])
print(nb_obs)
nb_rows = int(np.sqrt(nb_obs))
nb_cols = int(np.ceil(nb_obs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_obs:
            break
        axs[i, j].plot([obs[i * nb_cols + j] for obs in obses])
        axs[i, j].set_title(obses_names[i * nb_cols + j])

# set ylim between -5 and 5

for ax in axs.flat:
    ax.set_ylim([-5, 5])


fig.suptitle(f"{args.data}")
plt.show()
