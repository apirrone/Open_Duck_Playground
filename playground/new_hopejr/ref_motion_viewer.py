import mujoco
import numpy as np
import glfw
import time
import argparse
import os, sys
import pygame
from etils import epath
import mujoco.viewer
import pickle

# Import the reference motion class.
from playground.new_hopejr import base
from playground.common.poly_reference_motion import PolyReferenceMotion

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = f"{SCRIPT_PATH}/xmls"

COMMANDS_RANGE_X = [-0.3, 0.3]
COMMANDS_RANGE_Y = [-0.3, 0.3]
COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

# available_scenes = []
# if os.path.isdir(SCENE_PATH):
#     for name in os.listdir(SCENE_PATH):
#         if (
#             name.startswith("scene_")
#             and name.endswith(".xml")
#             and os.path.isfile(os.path.join(SCENE_PATH, name))
#         ):
#             scene_name = name[len("scene_") : -len(".xml")]
#             available_scenes.append(scene_name)
# if len(available_scenes) == 0:
#     print(f"No scenes found in: {SCENE_PATH}")
#     sys.exit(1)

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Reference Motion Viewer")
parser.add_argument(
    "--reference_data",
    type=str,
    default="playground/new_hopejr/data/polynomial_coefficients.pkl",
    help="Path to the polynomial coefficients pickle file.",
)
parser.add_argument(
    "-joystick", action="store_true", default=False, help="Use joystick control"
)
# Command parameters: dx, dy, dtheta
parser.add_argument(
    "--command",
    nargs=3,
    type=float,
    default=[0.0, -0.05, -0.1],
    help="Reference motion command as three floats: dx, dy, dtheta.",
)
# parser.add_argument(
#     "--scene",
#     type=str,
#     choices=available_scenes,
#     default="flat_terrain",
# )
args = parser.parse_args()
model_path = "playground/new_hopejr/xmls/scene.xml"

command = args.command

joystick1 = None
joystick2 = None
if args.joystick:
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick1 = pygame.joystick.Joystick(0)
        joystick1.init()
        command = [0.0, 0.0, 0.0]
        print("Joystick initialized:", joystick1.get_name())
        if pygame.joystick.get_count() > 1:
            joystick2 = pygame.joystick.Joystick(1)
            joystick2.init()
            print("Joystick 2 (theta) initialized:", joystick2.get_name())
        else:
            print(
                "Only one joystick detected; theta via second joystick will be disabled."
            )
    else:
        print("No joystick found!")

# Load the Mujoco model XML.
model = mujoco.MjModel.from_xml_string(
    epath.Path(model_path).read_text(), assets=base.get_assets()
)
data = mujoco.MjData(model)
model.opt.timestep = 0.002

# Step the simulation once to initialize.
mujoco.mj_step(model, data)

# Load the polynomial reference motion.
PRM = PolyReferenceMotion(args.reference_data)

# Get the "home" keyframe to use as a default pose.
home_frame = model.keyframe("home")
default_qpos = np.array(home_frame.qpos)
default_ctrl = np.array(home_frame.ctrl)
default_qpos[2] += 0.2  # Increase the base height by 0.2 meters

# Set initial state.
data.qpos[:] = default_qpos.copy()
data.ctrl[:] = default_ctrl.copy()


def key_callback(keycode):
    if joystick1 is not None:
        return

    print(f"key: {keycode}")
    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0
    if keycode == 265:  # arrow up
        lin_vel_x = COMMANDS_RANGE_X[1]
    if keycode == 264:  # arrow down
        lin_vel_x = COMMANDS_RANGE_X[0]
    if keycode == 263:  # arrow left
        lin_vel_y = COMMANDS_RANGE_Y[1]
    if keycode == 262:  # arrow right
        lin_vel_y = COMMANDS_RANGE_Y[0]
    if keycode == 81:  # a
        ang_vel = COMMANDS_RANGE_THETA[1]
    if keycode == 69:  # e
        ang_vel = COMMANDS_RANGE_THETA[0]

    command[0] = lin_vel_x
    command[1] = lin_vel_y
    command[2] = ang_vel
    print(f"command: {command}")


def handle_joystick():
    if joystick1 is None:
        return

    joy_z = 0
    pygame.event.pump()
    joy_y = joystick1.get_axis(1)
    joy_x = joystick1.get_axis(0)
    if joystick2 is not None:
        joy_z = joystick2.get_axis(0)
    if joy_y < 0:
        lin_vel_x = (-joy_y) * COMMANDS_RANGE_X[1]
    else:
        lin_vel_x = -joy_y * abs(COMMANDS_RANGE_X[0])
    lin_vel_y = -joy_x * COMMANDS_RANGE_Y[1]
    lin_vel_z = -joy_z * COMMANDS_RANGE_THETA[1]

    command[0] = lin_vel_x
    command[1] = lin_vel_y
    command[2] = lin_vel_z
    print(f"command: {command}")


# Create a Mujoco viewer to display the model.
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
) as viewer:
    step = 0
    dt = model.opt.timestep
    counter = 0
    new_qpos = default_qpos.copy()
    while viewer.is_running():
        step_start = time.time()
        handle_joystick()
        counter += 1
        new_qpos[:7] = default_qpos[:7].copy()
        if counter % 10 == 0:
            new_qpos = default_qpos.copy()
            if not all(val == 0.0 for val in command):
                imitation_i = step % PRM.nb_steps_in_period

                ref_motion = PRM.get_reference_motion(
                    command[0], command[1], command[2], imitation_i
                )
                ref_motion = np.array(ref_motion)
                # print(ref_motion.shape[0])
                if ref_motion.shape[0] == 62:
                    joints_pos = ref_motion[0:27]
                    ref_joint_pos = joints_pos
                    # ref_joint_pos = np.concatenate([joints_pos[:9], joints_pos[11:]])
                else:
                    print(
                        "Error: Unexpected reference motion dimension:",
                        ref_motion.shape,
                    )
                    sys.exit(1)
                # print(ref_joint_pos)
                # print(len(ref_joint_pos))
                # print(ref_joint_pos.shape)
                # exit()
                new_qpos = default_qpos.copy()
                if new_qpos[7 : 7 + 27].shape[0] == ref_joint_pos.shape[0]:
                    new_qpos[7 : 7 + 27] = ref_joint_pos
                else:
                    print(
                        "Error: Actuated joint dimension mismatch. Using default pose."
                    )
                step += 1
            else:
                step = 0
        data.qpos[:] = new_qpos

        # Step the simulation to update any dependent quantities.
        mujoco.mj_step(model, data)
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # step += 1
