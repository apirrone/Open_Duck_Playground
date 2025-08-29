import numpy as np
from typing import Optional
from playground.common.contact_aided_invariant_ekf.robot_kinematics import Kinematics

from pydrake.all import (
    MultibodyPlant,
    Parser,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    JacobianWrtVariable,
    StartMeshcat,
    MeshcatVisualizer
)


class DrakeKinematics:
    def __init__(
            self,
            imu_frame_name: str,
            end_effector_frame_name_to_id_map: dict[str],
            urdf_model_path: str,
            end_effector_offset_map: Optional[dict[str, list]] = None,
            visualize: bool = True
            ):
        self.end_effector_frame_name_to_id_map = end_effector_frame_name_to_id_map

        self.builder = DiagramBuilder()
        plant_and_scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.01)
        self.plant: MultibodyPlant = plant_and_scene_graph[0]
        self.scene_graph = plant_and_scene_graph[1]

        parser = Parser(self.plant)
        parser.AddModels(urdf_model_path)

        self.visualize = visualize
        if visualize:
            meshcat = StartMeshcat()
            MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, meshcat)

        self.imu_frame = self.plant.GetFrameByName(imu_frame_name)
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.imu_frame,
            RigidTransform()
        )
        self.plant.Finalize()
        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()

        self.end_effector_body_map = {}
        self.end_effector_frame_map = {}
        for name in end_effector_frame_name_to_id_map:
            self.end_effector_frame_map[name] = self.plant.GetFrameByName(name)
            self.end_effector_body_map[name] = self.plant.GetBodyByName(name)

        zero_offset = [0.0, 0.0, 0.0]
        if end_effector_offset_map is None:
            end_effector_offset_map = {name: zero_offset for name in end_effector_frame_name_to_id_map}
        self.end_effector_offset_map = end_effector_offset_map

        # Get the joint mapping
        self.joint_name_to_idx_map = {}
        for joint_idx in self.plant.GetJointIndices():
            joint = self.plant.get_joint(joint_idx)
            if joint.num_positions() == 1:
                self.joint_name_to_idx_map[joint.name()] = joint.position_start()

    def compute_fk(self, joint_positions: np.ndarray) -> dict[str, np.ndarray]:
        """
        Note: expect joint positions to be in the order of joing_name_to_idx_map
        """
        plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.plant.SetPositions(plant_context, joint_positions)
        if self.visualize:
            self.diagram.ForcedPublish(self.context)

        end_effector_pose_map = {}
        for name, body in self.end_effector_body_map.items():
            rigid_transform: RigidTransform = self.plant.EvalBodyPoseInWorld(
                plant_context,
                self.end_effector_body_map[name]
            )
            # Update the translation with the offset
            offset_rigid_transform = rigid_transform @ RigidTransform(
                self.end_effector_offset_map[name]
            )  # Apply in the body frame of the EE.
            end_effector_pose_map[name] = offset_rigid_transform.GetAsMatrix4()
        return end_effector_pose_map

    def compute_analytical_jacobians(self) -> dict[str, np.ndarray]:
        """
        Note: assumes `compute_fk` has been called to set the context.
        """
        plant_context = self.plant.GetMyContextFromRoot(self.context)
        # Computes spatial velocity (so(3)xR^3) from joint velocities of the foot w.r.t the imu frame.
        J_analyitical_map = {}

        for name, frame in self.end_effector_frame_map.items():
            J_analyitical_map[name] = self.plant.CalcJacobianSpatialVelocity(
                plant_context,
                JacobianWrtVariable.kQDot,
                frame,
                self.end_effector_offset_map[name],
                self.imu_frame,
                self.imu_frame,
            )
        return J_analyitical_map

    def compute_kinematics(self, joint_positions: np.ndarray, joint_covariance: np.ndarray) -> dict[str, Kinematics]:
        ee_transform_map = self.compute_fk(joint_positions=joint_positions)
        analytical_jacobian_map = self.compute_analytical_jacobians()

        kinematics_map = {}
        for ee_name in ee_transform_map:
            kinematic_covariance = analytical_jacobian_map[ee_name] @ joint_covariance @ analytical_jacobian_map[ee_name].T
            kinematics_map[ee_name] = Kinematics(
                id_in=self.end_effector_frame_name_to_id_map[ee_name],
                pose_in=ee_transform_map[ee_name],
                covariance_in=kinematic_covariance
            )
        return kinematics_map
