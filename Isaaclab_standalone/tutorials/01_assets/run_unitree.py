# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from get_all_prim_from_World import get_all_prim_from_World
##
# Pre-defined configs
##
from omni.isaac.lab_assets import UNITREE_A1_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)


    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)


    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    NITREE_cfg = UNITREE_A1_CFG.copy()
    NITREE_cfg.prim_path = "/World/Origin.*/Robot"
    NITREE = Articulation(cfg=NITREE_cfg)

    # return the scene information
    scene_entities = {"NITREE": NITREE}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["NITREE"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop reset
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            # root_state[:, :3] += origins
            # robot.write_root_state_to_sim(root_state)

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            print("--------------------------------------------------------","\n","root_state=",root_state)
#           root_state= position, quaternion, velocity, twist
#           root_state= tensor([[0.0000, 0.0000, 0.4200, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#           0.0000, 0.0000, 0.0000, 0.0000],
#           [0.0000, 0.0000, 0.4200, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#           0.0000, 0.0000, 0.0000, 0.0000]])
# /World/Origin1/Robot/trunk is the articulation
            print("--------------------------------------------------------","\n","joint_pos=",joint_pos)
#  joint_pos= tensor([[ 0.1889, -0.0618,  0.1992, -0.0749,  0.8653,  0.8700,  1.0955,  1.0461,
#          -1.4166, -1.4994, -1.4037, -1.4698],
#         [ 0.1155, -0.0710,  0.1271, -0.0651,  0.8274,  0.8508,  1.0336,  1.0674,
#          -1.4766, -1.4472, -1.4055, -1.4559]])

# /World/Origin1/Robot/trunk/FL_hip_joint
# /World/Origin1/Robot/trunk/FR_hip_joint
# /World/Origin1/Robot/trunk/RL_hip_joint
# /World/Origin1/Robot/trunk/RR_hip_joint
# /World/Origin1/Robot/FL_hip/FL_thigh_joint
# /World/Origin1/Robot/FL_thigh/FL_calf_joint
# /World/Origin1/Robot/FR_hip/FR_thigh_joint
# /World/Origin1/Robot/FR_thigh/FR_calf_joint
# /World/Origin1/Robot/RL_hip/RL_thigh_joint
# /World/Origin1/Robot/RL_thigh/RL_calf_joint
# /World/Origin1/Robot/RR_hip/RR_thigh_joint
# /World/Origin1/Robot/RR_thigh/RR_calf_joint

# [ FL_hip, FL_hip, RR_hip , RR_hip,  FL_thigh, FL_thigh, RR_thigh , RR_thigh,
#         FL_calf, FL_calf, RR_calf , RR_calf]

    #     init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.42),
    #     joint_pos={
    #         ".*L_hip_joint": 0.1,
    #         ".*R_hip_joint": -0.1,
    #         "F[L,R]_thigh_joint": 0.8,
    #         "R[L,R]_thigh_joint": 1.0,
    #         ".*_calf_joint": -1.5,
    #     }, 一共 12个joint
    #     joint_vel={".*": 0.0},
    # ), 
            print("--------------------------------------------------------","\n","joint_vel=",joint_vel)

#  joint_pos= tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")

        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # get and print all primes
    all_prim_list = get_all_prim_from_World()

    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
