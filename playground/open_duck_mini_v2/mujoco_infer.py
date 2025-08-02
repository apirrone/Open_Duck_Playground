import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from playground.open_duck_mini_v2.command_handler import CommandHandler, CommandLimits

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool,
        use_api: bool = False
    ):
        super().__init__(model_path)

        self.standing = standing
        self.use_api = use_api

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

        # Initialize command handler with limits
        limits = CommandLimits(
            linear_velocity_x=(-0.15, 0.15),
            linear_velocity_y=(-0.2, 0.2),
            angular_velocity=(-1.0, 1.0),
            neck_pitch=(-0.34, 1.1),
            head_pitch=(-0.78, 0.78),
            head_yaw=(-1.5, 1.5),
            head_roll=(-0.5, 0.5)
        )
        
        if self.use_api:
            # Use the global command handler from API server
            from playground.open_duck_mini_v2.api_server import get_command_handler
            self.command_handler = get_command_handler()
        else:
            # Create local command handler for keyboard control
            self.command_handler = CommandHandler(limits)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        # Commands are now managed by command_handler

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")
        
        # Selection tracking for red highlighting
        self.last_selected_body = -1
        self.original_geom_colors = {}

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

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

    def key_callback(self, keycode):
        if not self.use_api:
            print(f"key: {keycode}")
            if not self.command_handler.process_keyboard_input(keycode):
                # Handle phase frequency changes
                if keycode == 80:  # p
                    self.phase_frequency_factor += 0.1
                elif keycode == 59:  # m
                    self.phase_frequency_factor -= 0.1
    
    def update_selection_highlight(self, viewer):
        """Update red highlighting based on current selection."""
        current_selected = viewer.perturb.select
        
        if current_selected != self.last_selected_body:
            # Restore previous selection's colors
            if self.last_selected_body > 0:
                for geom_id, original_color in self.original_geom_colors.items():
                    self.model.geom_rgba[geom_id] = original_color
                self.original_geom_colors.clear()
            
            # Highlight new selection
            if current_selected > 0:
                for i in range(self.model.ngeom):
                    if self.model.geom_bodyid[i] == current_selected:
                        # Store original color
                        self.original_geom_colors[i] = self.model.geom_rgba[i].copy()
                        # Set to red
                        self.model.geom_rgba[i] = [1.0, 0.0, 0.0, 1.0]
            
            self.last_selected_body = current_selected

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
                    time.sleep(0.001)  # Yield GIL to allow API server to run
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
                            self.command_handler.get_commands(),
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
                    
                    # Update selection highlighting
                    self.update_selection_highlight(viewer)

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
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument("--use_api", action="store_true", default=False,
                        help="Use API server for control instead of keyboard")

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing,
        args.use_api
    )
    
    mjinfer.run()
