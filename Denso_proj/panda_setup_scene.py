import argparse
import os
import time
from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on building scene of the franka robot arm.")

parser.add_argument("--robot", type=str, default="franka", help="Name of the robot.")


parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# parser.add_argument("--headless", type=bool, default=True, help="")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""


import torch
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
from omni.isaac.core.articulations import ArticulationView


##
# Pre-defined configs
##
from franka import FRANKA_PANDA_HIGH_PD_CFG

# create table scene config
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
    if args_cli.robot == "franka":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print(robot)
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported.")


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


    print("scene articulation:",scene.articulations)
    print("scene namespace:",scene.env_ns)
    # /World/envs
    print("scene namespace:",scene.env_origins)
    #tensor([[ 1.,  0.,  0.], [-1.,  0.,  0.]], device='cuda:0')
    print("scene env_prim_paths:",scene.env_prim_paths)
    # ['/World/envs/env_0', '/World/envs/env_1']
    print("scene keys:",scene.keys())

    for attr_name, attr_value in scene.__dict__.items():
        print(f"{attr_name} = {attr_value}")
    
    robot = scene["robot"]
    # <omni.isaac.lab.assets.articulation.articulation.Articulation object at 0x000002D2AFFD0B50>

    print(robot)
    for attr_name, attr_value in robot.__dict__.items():
        print(f"{attr_name} = {attr_value}")

          
    if args_cli.robot == "franka":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint.*"], body_names=["link6"])
        # "link6" are of interest within this entity configuration
        # typically the end-effector link
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported.")


    # Resolving the scene entities
    # robot_entity_cfg.resolve(scene)

    # print(robot_entity_cfg.object_collection_names)

    # print(robot_entity_cfg.object_collection_ids)


    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
