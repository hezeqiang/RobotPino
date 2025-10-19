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
    # python.exe c:/Users/13306/OneDrive/Coding_Proj/issaaclab140/RobotPino/Denso_proj/panda_run_diff_ik_pino.py

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
from RobotModel import RobotModel as RobotModel
from omni.isaac.lab.assets.articulation.articulation import Articulation
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from utili.tool_quaternion_ope import posquat_to_se3, se3_to_posquat
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns,ImuCfg
##
# Pre-defined configs
##
from franka import FRANKA_PANDA_CFG
from omni.isaac.core.objects import DynamicCuboid
from scipy.spatial.transform import Rotation
import logging

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

    # movable wall
    wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall",
        spawn=sim_utils.CuboidCfg(  # Define the wall as a dynamic cuboid
            size=(1.2, 1.0, 2),  # Wall dimensions: 0.1m thick, 1m wide, 0.5m tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # rigid_body_enabled=True,  # The wall is a rigid body
                kinematic_enabled=False,  # The wall is kinematic
                disable_gravity=False, # Enable gravity
                # enable_gyroscopic_forces=True,
            ),
            physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction = 0.1,
                dynamic_friction = 0.1,
            ),
            mass_props= sim_utils.MassPropertiesCfg(mass=1),
            collision_props= sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material= sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.8, opacity=0.9),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0),  # Position: 0.3m in front of the table, aligned with the ground
            rot=(0.0, 0.0, 0.0, 1.0),  # No rotation
        ),
    )

    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_hand", update_period=0.0, history_length=6, debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/wall"],
    #     # filter of external contact forces and keep the wall force only
    # )

    # imu_hand = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand", debug_vis=True)
    # imu_hand_without_gravity = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand",gravity_bias=(0, 0, 0), debug_vis=True)


    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


    # Tilted wall
    # tilted_wall = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/TiltedWall",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(2.0, 1.5, 0.01),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         activate_contact_sensors=True,
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.6 + 0.085, 0.0, 0.3), rot=(0.9238795325, 0.0, -0.3826834324, 0.0)
    #     ),
    # )

    # contact_forces = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/TiltedWall",
    #     update_period=0.0,
    #     history_length=2,
    #     debug_vis=False,
    # )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot:Articulation, logger):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    # robot = scene["robot"]
    print("robot: ",robot,"\n") # >>> Robot

    robotpin = Init_robotpin()
    spawn_robotpin = [robotpin]*scene.num_envs

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals_quat = np.array( [
        [0.5, 0, 0.5, 0, 1, 0, 0],  # z-axis pointing to ground (180° rotation around y-axis)
        [0.2, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
    ]) # [pos, quat]

    # Convert quaternion to se3
    ee_goals_se3 = posquat_to_se3(ee_goals_quat)

    # Track the given command
    current_goal_idx = 0
    # Create spawn buffers to store actions
    spawn_ee_goals_se3 = np.zeros((scene.num_envs, 6))
    spawn_ee_pose_Qauat_w_numpy = np.zeros((scene.num_envs, 7))
    spawn_ee_goals_quat = np.zeros((scene.num_envs, 7))
    spawn_ee_goals_se3[:] = ee_goals_se3[current_goal_idx]

    
    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=[spawn_robotpin[0].ee_frame_name]) # end-effector frame = panda_hand
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # print(robot_entity_cfg.body_ids) # [7]
    # print(robot_entity_cfg.body_names) # 'panda_hand' end-effector

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

    q_guess = np.array([0, -0.569, 0, -2.810, 0, 3.037, 0.741, 0.04, 0.04]) 

    Simu_count = 300
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % Simu_count == 0:
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals_quat)

            # set goal
            spawn_ee_goals_se3[:] = ee_goals_se3[current_goal_idx]
            spawn_tua_command = np.zeros((scene.num_envs, spawn_robotpin[0].model.nv))

            ee_goals_SE3 = RobotModel.se3_to_SE3(spawn_ee_goals_se3[0])

            print(spawn_robotpin[0].ee_frame_id) # 20 panda_hand
            q_des = spawn_robotpin[0].controller.IK_step(spawn_robotpin[0].ee_frame_id, ee_goals_SE3, q_guess)

            print("Goal SE3 config:")
            spawn_robotpin[0].print_formatted_nd_matrix(ee_goals_SE3.homogeneous)

            pin.forwardKinematics(spawn_robotpin[0].model, spawn_robotpin[0].data, q_des)
            pin.updateFramePlacements(spawn_robotpin[0].model, spawn_robotpin[0].data)

            # Get current end-effector pose
            desired_pose = spawn_robotpin[0].data.oMf[spawn_robotpin[0].ee_frame_id]

            print("IK calculated desired SE3 config:")
            spawn_robotpin[0].print_formatted_nd_matrix(desired_pose.homogeneous)
            print("Desired joint position target:" , q_des) #  the result is right while neglect the collision and 7 DoFs

            # the data read from the body_link_state_w is not consistent with the data by joint angle, due to the collision of ee frame definitions. 
            # we believe in the joint angle instead of body_link_state_w
            spawn_ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0]].detach().cpu().numpy() # world frame [pos, quat, lin_vel, ang_vel]
            spawn_root_pose_w = robot.data.root_link_state_w[:].detach().cpu().numpy()

            for i in range(scene.num_envs):
                
                # print(spawn_robotpin[i].controller)   
                spawn_robotpin[i].controller.reset()

                # set the target pose
                spawn_robotpin[i].controller.q_des = q_des                


            # print(ee_pose_se3_w[0:3]-root_pose_se3_w[0:3], ee_goals_quat[current_goal_idx,0:3])

            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # reset actions            
            robot.reset()

        # else:
        # obtain quantities from simulation

        # read q and dq, ddq with encoder
        spawn_q = robot.data.joint_pos.detach().detach().cpu().numpy()
        spawn_dq = robot.data.joint_vel.detach().detach().cpu().numpy()
        spawn_ddq = robot.data.joint_acc.detach().cpu().numpy()

        if (count) % Simu_count == Simu_count-1:
            spawn_robotpin[0].print_formatted_nd_matrix(ee_pose_SE3_w.homogeneous)
            print("Real joint position:", spawn_q[0])
            # print("Final real SE3 config::", ee_pose_SE3_re)
            print("Desired joint position target:" , q_des)
            # print("q_error:",spawn_robotpin[0].controller.q_error)
            # print("Real joint v target:", spawn_dq[0])
            # print("Real joint a target:", spawn_ddq[0])
            # print("tau_bias:", spawn_robotpin[0].controller.tau_bias)
            # print("tau_error:",spawn_robotpin[0].controller.tau_error)
            # print("tua_command:", spawn_tua_command[0])
            # print("root_pose_SE3_w",root_pose_SE3_w)
            # print("ee_pose_SE3_w", ee_pose_SE3_w)

            # print("ee_pose_quat_tensor[0:3]",spawn_ee_pose_quat_tensor[3, 0:3])

            # print(spawn_tua_command_tensor)
            # print(robot_entity_cfg.joint_ids)
            # print(spawn_tua_command)
            # print(ee_pose_SE3_re)

        for i in range(scene.num_envs):

            pin.forwardKinematics(spawn_robotpin[i].model, spawn_robotpin[i].data, spawn_q[i])
            pin.updateFramePlacements(spawn_robotpin[i].model, spawn_robotpin[i].data)
            
            ee_pose_SE3_w = spawn_robotpin[i].data.oMf[spawn_robotpin[0].ee_frame_id]

            root_pose_se3_w = posquat_to_se3(spawn_root_pose_w[i, 0:7])
            root_pose_SE3_w = RobotModel.se3_to_SE3(root_pose_se3_w)

            # ee_pose_se3_w = RobotModel.SE3Tose3(root_pose_SE3_w * ee_pose_SE3_w ) # evn_num * 6 [q,w]
            # se3_twist = pin.log6(ee_pose_SE3_w).vector  # Returns a 6D numpy array

            spawn_ee_pose_Qauat_w_numpy[i] = RobotModel.SE3ToQuat(ee_pose_SE3_w)
            
            # print(q_des, spawn_q[i] , spawn_dq[i], spawn_ddq[i])
            # compute the tua list of each joint
            spawn_robotpin[i].controller.ID_step(q_des, spawn_q[i] , spawn_dq[i], spawn_ddq[i])

            spawn_tua_command[i] = spawn_robotpin[i].controller.tua_command[0:9]
            # print("tua_command:", spawn_tua_command[i])

            # logger
            if i==0:
                # logger.info(f"spawn_q[{i}]: {spawn_q[0]}, translation: {ee_pose_SE3_w.translation.tolist()}, rotation: {ee_pose_SE3_w.rotation.tolist()}")
                logger.info(f"spawn_q[{i}]: {spawn_q[0]}")
  
            # default_q = np.array([0, -0.569, 0, -2.810, 0, 3.037, 0.741, 0.04, 0.04]) 
            # joint_pos_des = torch.tensor(np.array([default_q[0:7]] * scene.num_envs), dtype=torch.float32, device=sim.device)

        spawn_tua_command_tensor = torch.from_numpy(spawn_tua_command).to(torch.float32).to(device=sim.device)

        # apply actions
        # robot.set_joint_effort_target(spawn_tua_command_tensor, joint_ids=robot_entity_cfg.joint_ids) # joint_ids = [0,1,...6]
        robot.set_joint_effort_target(spawn_tua_command_tensor)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        spawn_ee_pose_quat_tensor = torch.from_numpy(spawn_ee_pose_Qauat_w_numpy).to(torch.float32).to(device=sim.device)
        # update marker positions
        ee_marker.visualize(spawn_ee_pose_quat_tensor[:, 0:3] + scene.env_origins, spawn_ee_pose_quat_tensor[:, 3:7])

        # convert spawn_ee_goals_se3 to quaternion
        # spawn_ee_goals_quat[:] = ee_goals_quat[current_goal_idx]
        spawn_ee_goals_quat = se3_to_posquat(spawn_ee_goals_se3)
        spawn_ee_goals_quat_tensor = torch.from_numpy(spawn_ee_goals_quat ).to(torch.float32).to(device=sim.device)

        goal_marker.visualize(spawn_ee_goals_quat_tensor[:, 0:3] + scene.env_origins, spawn_ee_goals_quat_tensor[:, 3:7])


def Init_robotpin() -> RobotModel:
    panda_robot_urdf_file =r"C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\RobotPino\franka_description\robots\panda_arm_hand.urdf"
    package_dirs=r"C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\RobotPino"

    # with open (panda_robot_urdf_file, "r") as file:
    #     robot_urdf = file.read()

    # Initialize robot model
    # config the root frame, the ee body frame()
    config={
                "ee_frame": "panda_hand",
                "root_frame": "panda_link0",
                "namespace": "panda", 
            }
    
    robot = RobotModel(config)

    robot.load_desc_from_file(panda_robot_urdf_file, package_dirs)
    robot.init_pin_model()

    return robot

def main():

    # Set up a logger for this class
    logger_tra = logging.getLogger(__name__)
    logger_tra.setLevel(logging.DEBUG)
    # Create a file handler that logs even debug messages
    fh = logging.FileHandler(r'RobotPino\Denso_proj\Trajectory.log', mode='w')
    fh.setLevel(logging.DEBUG)
    # Create a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the file handler to the logger
    logger_tra.addHandler(fh)


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
    run_simulator(sim, scene, robot, logger_tra)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
