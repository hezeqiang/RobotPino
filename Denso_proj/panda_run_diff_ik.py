# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app and then import 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import pinocchio as pin
from omni.isaac.lab.assets.articulation.articulation import Articulation
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from utili.tool_quaternion_ope import posquat_to_se3
##
# Pre-defined configs
##
from franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

def se3_to_SE3(vec): #[q,w]
    """
    Convert a 6D vector (translation + rotation vector) into a pinocchio.SE3 object.
    
    Parameters:
        vec: np.array of shape (6,), where vec[:3] is translation and vec[3:] is a rotation vector.
    
    Returns:
        se3: a pinocchio.SE3 object.
    """
    # Extract translation and rotation vector
    translation = vec[:3]
    rot_vec = vec[3:]
    
    # Convert the rotation vector (in so(3)) to a rotation matrix using the exponential map.
    # Note: pin.SO3.Exp returns an SO3 object, and we take its matrix() for a 3x3 rotation matrix.
    R = pin.exp3(rot_vec)
    
    return pin.SE3(R, translation)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot:Articulation):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    # robot = scene["robot"]
    print("robot: ",robot,"\n") # >>> Robot


    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.2, 0.3, 0.7, 0.707, 0, 0.707, 0],
        [0.2, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.2, 0, 0.3, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_link7"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    print(robot_entity_cfg.body_ids) # [7]
    print(robot_entity_cfg.body_names) # 'panda_hand' end-effector

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] 


    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:

            ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_link_state_w[:, 0:7]
            print("ee_pose_w: ",ee_pose_w,"\n",
                  "root_pose_w: ",root_pose_w,"\n"
                  "relateve_ee_pose_w: ",ee_pose_w[:,0:3]-root_pose_w[:,0:3],"\n"
                  "ik_commands: ",ik_commands[:,0:3],"\n")
    

            ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:7].detach().cpu().numpy() # world frame
            root_pose_w = robot.data.root_link_state_w[:, 0:7].detach().cpu().numpy()
            ee_pose_se3_w = posquat_to_se3(ee_pose_w[0]) # evn_num * 6 [q,w]
            root_pose_se3_w = posquat_to_se3(root_pose_w[0])

            ee_pose_SE3_w = se3_to_SE3(ee_pose_se3_w)
            root_pose_SE3_w = se3_to_SE3(root_pose_se3_w)

            root_pose_SE3_w_inv = root_pose_SE3_w.inverse()
            ee_pose_SE3_re = root_pose_SE3_w_inv * ee_pose_SE3_w

            # print(ee_pose_SE3_re)

            print("Desired joint position target:" , robot.data.joint_pos_target)
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_link_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # default_q = np.array([0, -0.569, 0, -2.810, 0, 3.037, 0.741, 0.04, 0.04]) 
            # default_q = np.array([0, 0, 0, 0, 0, 3.037, 0, 0, 0]) 
            # joint_pos_des = torch.tensor(np.array([default_q[0:7]] * scene.num_envs), dtype=torch.float, device=sim.device)
            # print(joint_pos_des)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)


        # obtain quantities from simulation
        ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    
    print("scene_cfg: ",scene_cfg.robot,"\n")

    scene = InteractiveScene(scene_cfg)

    print("scene",scene,"\n") # >>> Robot
    print("scene articulation:",scene.articulations["robot"].cfg,"\n") # >>> Robot cfg
    # print("scene articulation actuators:", scene.articulations["robot"].actuators,"\n") # >>> error code
    print("scene namespace:",scene.env_ns,"\n") # >>> /World/envs
    # /World/envs
    print("scene env origins:",scene.env_origins,"\n") # >>>([[ 1.,  0.,  0.], [-1.,  0.,  0.]], device='cuda:0')
    #tensor([[ 1.,  0.,  0.], [-1.,  0.,  0.]], device='cuda:0')
    print("scene env_prim_paths:",scene.env_prim_paths,"\n") # >>> ['/World/envs/env_0', '/World/envs/env_1']
    # ['/World/envs/env_0', '/World/envs/env_1']
    print("scene keys:",scene.keys(),"\n") # >>> ['terrain', 'robot', 'ground', 'dome_light', 'table']


    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    robot = scene["robot"]
    # Run the simulator
    run_simulator(sim, scene,robot)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
