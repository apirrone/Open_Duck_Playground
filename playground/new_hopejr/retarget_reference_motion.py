import jax.numpy as jp
import json
from scipy.spatial.transform import Rotation as R

# [
#    	duration of frame in seconds (1D),
#    	root position (3D),
#    	root rotation (4D),
#    	chest rotation (4D),
#    	neck rotation (4D),
#    	right hip rotation (4D),
#    	right knee rotation (1D),
#    	right ankle rotation (4D),
#    	right shoulder rotation (4D),
#    	right elbow rotation (1D),
#    	left hip rotation (4D),
#    	left knee rotation (1D),
#    	left ankle rotation (4D),
#    	left shoulder rotation (4D),
#    	left elbow rotation (1D)
# ]


class RetargetReferenceMotion:
    def __init__(self, reference_file: str):
        file = open(reference_file, "r")
        data = json.load(file)
        self.frames = data["Frames"]
        self.hopejr_dofs_names = [
            "torso_yaw",
            "torso_roll",
            "torso_pitch",
            "right_hip_pitch",
            "right_hip_roll",
            "right_hip_yaw",
            "right_knee",
            "right_ankle_pitch",
            "right_ankle_roll",
            "right_toe",
            "left_hip_pitch",
            "left_hip_roll",
            "left_hip_yaw",
            "left_knee",
            "left_ankle_pitch",
            "left_ankle_roll",
            "left_toe",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_elbow_yaw",
            "right_elbow_pitch",
            "right_forearm_yaw",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_elbow_yaw",
            "left_elbow_pitch",
            "left_forearm_yaw",
        ]

        # self.target_fps = 100.0
        self.target_fps = 50.0
        self.data_fps = int(1.0 / self.frames[0][0])
        self.converted_frames = self.convert_to_hopejr()
        self.converted_frames = self.interpolate_frames(self.converted_frames)
        self.converted_frames = jp.array(self.converted_frames)
        self.nb_steps_in_period = len(self.converted_frames)

    def interpolate_frames(self, converted_frames):
        # Add or remove frames to match the target FPS
        target_nb_frames = int(
            self.target_fps * (len(converted_frames) / self.data_fps)
        )
        if target_nb_frames == len(converted_frames):
            return converted_frames
        elif target_nb_frames < len(converted_frames):
            # Remove frames
            step = len(converted_frames) // target_nb_frames
            return converted_frames[::step]
        else:
            # Add frames by linear interpolation
            new_frames = []
            for i in range(target_nb_frames):
                index = i * len(converted_frames) // target_nb_frames
                if index < len(converted_frames) - 1:
                    next_index = index + 1
                    ratio = (i * len(converted_frames)) / target_nb_frames - index
                    new_frame = [
                        (1 - ratio) * converted_frames[index][j]
                        + ratio * converted_frames[next_index][j]
                        for j in range(len(converted_frames[0]))
                    ]
                    new_frames.append(new_frame)
            return new_frames

    def sample_frame(self, i):
        return jp.array(self.converted_frames[i])

    def parse_frame(self, frame):
        duration = frame[0]
        root_pos = frame[1:4]
        root_quat = frame[4:8]
        chest_quat = frame[8:12]
        neck_quat = frame[12:16]
        right_hip_quat = frame[16:20]
        right_knee_angle = frame[20]
        right_ankle_quat = frame[21:25]
        right_shoulder_quat = frame[25:29]
        right_elbow_angle = frame[29]
        left_hip_quat = frame[30:34]
        left_knee_angle = frame[34]
        left_ankle_quat = frame[35:39]
        left_shoulder_quat = frame[39:43]
        left_elbow_angle = frame[43]

        return {
            "duration": duration,
            "root_pos": root_pos,
            "root_quat": root_quat,
            "chest_quat": chest_quat,
            "neck_quat": neck_quat,
            "right_hip_quat": right_hip_quat,
            "right_knee_angle": right_knee_angle,
            "right_ankle_quat": right_ankle_quat,
            "right_shoulder_quat": right_shoulder_quat,
            "right_elbow_angle": right_elbow_angle,
            "left_hip_quat": left_hip_quat,
            "left_knee_angle": left_knee_angle,
            "left_ankle_quat": left_ankle_quat,
            "left_shoulder_quat": left_shoulder_quat,
            "left_elbow_angle": left_elbow_angle,
        }

    def compute_joints_vels(self, joints_positions, dts):
        joints_vels = []
        for i in range(len(joints_positions) - 1):
            vel = list(
                (jp.array(joints_positions[i + 1]) - jp.array(joints_positions[i]))
                / dts[i]
            )
            joints_vels.append(vel)
        # Append the last velocity as zero since we don't have a next frame
        joints_vels.append(list(jp.zeros_like(jp.array(joints_positions[0]))))
        return joints_vels

    def compute_base_linear_vel(self, root_positions, dts):
        base_linear_vel = []
        for i in range(len(root_positions) - 1):
            vel = list(
                (jp.array(root_positions[i + 1]) - jp.array(root_positions[i])) / dts[i]
            )
            base_linear_vel.append(vel)
        # Append the last velocity the same as the last position
        base_linear_vel.append(base_linear_vel[-1].copy())
        return jp.array(base_linear_vel)

    def compute_base_angular_vel(self, root_quats, dts):
        # euler
        base_angular_vel = []
        for i in range(len(root_quats) - 1):
            quat1 = R.from_quat(root_quats[i])
            quat2 = R.from_quat(root_quats[i + 1])
            delta_quat = quat2 * quat1.inv()
            euler_vel = delta_quat.as_euler("xyz", degrees=False) / dts[i]
            base_angular_vel.append(jp.array(euler_vel))

        # Append the last velocity the same as the last position
        base_angular_vel.append(base_angular_vel[-1].copy())

        return jp.array(base_angular_vel)

    def convert_to_hopejr(self):
        # output = [
        #     joints_pos, # 27
        #     joints_vel, # 27
        #     foot_contacts, # 2
        #     base_linear_vel, # 3
        #     base_angular_vel # 3
        # ]
        joints_positions = []
        root_positions = []
        root_quats = []
        dts = []
        for frame in self.frames:
            parsed_frame = self.parse_frame(frame)
            joints_positions.append(self.get_frame_joints_pos(parsed_frame))
            dts.append(parsed_frame["duration"])
            root_positions.append(parsed_frame["root_pos"])
            root_quats.append(parsed_frame["root_quat"])

        joints_vels = self.compute_joints_vels(joints_positions, dts)
        foot_contacts = jp.zeros((len(self.frames), 2))  # Placeholder for foot contacts
        base_linear_vel = self.compute_base_linear_vel(root_positions, dts)
        base_angular_vel = self.compute_base_angular_vel(root_quats, dts)

        # The output structure is a list of numpy arrays, each corresponding to a different part of the state

        converted_frames = []
        for i in range(len(self.frames)):
            converted_frame = jp.concatenate(
                [
                    jp.array(joints_positions[i]),
                    jp.array(joints_vels[i]),
                    jp.array(foot_contacts[i]),
                    jp.array(base_linear_vel[i]),
                    jp.array(base_angular_vel[i]),
                ]
            )

            converted_frames.append(list(converted_frame))
        return converted_frames

    def get_frame_joints_pos(self, parsed_frame: dict):
        torso_euler = R.from_quat(parsed_frame["chest_quat"]).as_euler(
            "xyz", degrees=False
        )
        torso_roll = -torso_euler[0] + jp.pi
        torso_pitch = torso_euler[1]
        torso_yaw = torso_euler[2]

        right_hip_euler = R.from_quat(
            parsed_frame["right_hip_quat"], scalar_first=True
        ).as_euler("xyz", degrees=False)
        right_hip_roll = right_hip_euler[0]
        right_hip_pitch = right_hip_euler[1]
        # right_hip_yaw = right_hip_euler[2]
        right_hip_yaw = 0

        right_knee = parsed_frame["right_knee_angle"]

        right_ankle_euler = R.from_quat(
            parsed_frame["right_ankle_quat"], scalar_first=True
        ).as_euler("xyz", degrees=False)
        # right_ankle_roll = right_ankle_euler[0]
        right_ankle_roll = right_ankle_euler[0]
        right_ankle_pitch = right_ankle_euler[1]
        right_toe = 0.0

        left_hip_euler = R.from_quat(
            parsed_frame["left_hip_quat"], scalar_first=True
        ).as_euler("xyz", degrees=False)
        left_hip_roll = -left_hip_euler[0]
        left_hip_pitch = -left_hip_euler[1]
        # left_hip_yaw = left_hip_euler[2]
        left_hip_yaw = 0

        left_knee = -parsed_frame["left_knee_angle"]

        left_ankle_euler = R.from_quat(
            parsed_frame["left_ankle_quat"], scalar_first=True
        ).as_euler("xyz", degrees=False)
        left_ankle_roll = left_ankle_euler[0]
        left_ankle_pitch = left_ankle_euler[1]
        left_toe = 0.0

        right_shoulder_euler = R.from_quat(
            parsed_frame["right_shoulder_quat"]
        ).as_euler("xyz", degrees=False)
        right_shoulder_roll = -right_shoulder_euler[0] - jp.pi
        right_shoulder_pitch = -right_shoulder_euler[1]

        right_elbow_pitch = parsed_frame["right_elbow_angle"]
        right_elbow_yaw = 0.0
        right_forearm_yaw = 0.0

        left_shoulder_euler = R.from_quat(parsed_frame["left_shoulder_quat"]).as_euler(
            "xyz", degrees=False
        )
        left_shoulder_roll = -left_shoulder_euler[0] - jp.pi
        left_shoulder_pitch = -left_shoulder_euler[1]

        left_elbow_pitch = -parsed_frame["left_elbow_angle"]
        left_elbow_yaw = 0.0
        left_forearm_yaw = 0.0

        joints_pos = [
            torso_yaw,
            torso_roll,
            torso_pitch,
            right_hip_pitch,
            right_hip_roll,
            right_hip_yaw,
            right_knee,
            right_ankle_pitch,
            right_ankle_roll,
            right_toe,
            left_hip_pitch,
            left_hip_roll,
            left_hip_yaw,
            left_knee,
            left_ankle_pitch,
            left_ankle_roll,
            left_toe,
            right_shoulder_pitch,
            right_shoulder_roll,
            right_elbow_yaw,
            right_elbow_pitch,
            right_forearm_yaw,
            left_shoulder_pitch,
            left_shoulder_roll,
            left_elbow_yaw,
            left_elbow_pitch,
            left_forearm_yaw,
        ]
        return joints_pos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retarget Reference Motion")
    parser.add_argument(
        "--reference_file",
        type=str,
        default="playground/new_hopejr/data/reference_motion.json",
        help="Path to the reference motion JSON file.",
    )
    args = parser.parse_args()

    retarget_motion = RetargetReferenceMotion(args.reference_file)
