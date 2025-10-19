from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.articulations import Articulation
from typing import Optional



class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:

        # change the config path
        yaml_path = "C:/isaacsim/lib/isaac-sim-4.0.0/standalone_examples/MyStandalone/rmpflow/robot_descriptor.yaml"
        urdf_path = "C:/Users/13306/OneDrive/Isaac/cobotta_pro_900/cobotta_pro_900.urdf"
        
        self._kinematics = LulaKinematicsSolver(robot_description_path = yaml_path,
                                                urdf_path = urdf_path)
        
        #TODO: change the config path
        if end_effector_frame_name is None:
            end_effector_frame_name = "onrobot_rg6_base_link"
        
        # Call the parent class constructor to initialize the kinematics solver for the given robot articulation and end-effector frame
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)

        return