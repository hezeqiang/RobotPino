import argparse

from omni.isaac.lab.app import AppLauncher
from quaternion_ope import quat_tensor_to_rot_matrix
import modern_robotics as MR

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on building scene of the r068 robot arm.")

parser.add_argument("--robot", type=str, default="r068", help="Name of the robot.")


parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""


import torch
import get_all_prim_from_World
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

##
# Pre-defined configs
##
from r068 import R068_CFG_HIGH_PD_CFG


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
    if args_cli.robot == "r068":
        robot = R068_CFG_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print(robot)
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported.")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    
    print(robot)

    # Create controller
    # configuration
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    print(diff_ik_cfg)
    # create controller instance
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    print(diff_ik_controller)

    # Markers of the current and target end-effector poses
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    # screw? quant?
    ee_goals = [
        [0.2, 0.3, 0.7, 0.707, 0, 0.707, 0],
        [0.3, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.3, 0, 0.6, 0.0, 1.0, 0.0, 0.0],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]


    # Obtain the robot entity configuration
    # Specify robot-specific parameters
    if args_cli.robot == "r068":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint.*"], body_names=["link6"])
        # "link6" are of interest within this entity configuration
        # typically the end-effector link
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported.")

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    print(robot_entity_cfg.body_ids) # >>> [6]
    print(robot_entity_cfg.body_names) # >>> link6

    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 # >>> 5
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] # >>> 6

    print(ee_jacobi_idx) # >>> 6

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
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

            
            CoM_pos_link = robot.data.com_pos_b
            CoM_quat_link = robot.data.com_quat_b

            # ee-> end-effector
            # root -> root frame
            # the data will be like [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0]
            ee_pose_w_tensor = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:13]
            # first ":" means all envs, "second" means the first body_id, "third" means the State of all bodies [pos, quat, lin_vel, ang_vel] in simulation world frame.
            # format: tensor [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            ee_pose_w_numpy = ee_pose_w_tensor.detach().cpu().numpy()

            pos=ee_pose_w_numpy[:,0:3]
            quat=ee_pose_w_numpy[:,3:7]
            lin_vel=ee_pose_w_numpy[:,7:10]
            ang_vel=ee_pose_w_numpy[:,10:13] 
            print("pos : ", pos,
                "quat : ", quat,
                "lin_vel : ", lin_vel,
                "ang_vel : ", ang_vel)


            root_pose_w = robot.data.root_link_state_w[:, 0:7] # root id = 0
            root_pose_w_numpy = root_pose_w.detach().cpu().numpy()
            root_pos=root_pose_w_numpy[:,0:3]
            root_quat=root_pose_w_numpy[:,3:7]
            print("root_pos : ", root_pos,
                "root_quat : ", root_quat)


            # compute frame in root frame

            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )

            # compute the joint commands
            # computes the target joint positions that will yield the desired end effector pose.
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)



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

    get_all_prim_from_World.get_all_prim_from_World()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
