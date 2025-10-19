import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation

# a class of reactive motion policies designed to parameterize non-Euclidean behaviors as dynamical systems 
class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:
        # TODO: chamge the follow paths

        self.rmpflow = mg.lula.motion_policies.RmpFlow(robot_description_path=r"c:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\IsaacSim\standalone_examples\MyStandalone\rmpflow\robot_descriptor.yaml",
                                                        rmpflow_config_path=r"c:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\IsaacSim\standalone_examples\MyStandalone\controllers\rmpflow.py",
                                                        urdf_path=r"C:\Users\13306\OneDrive\Isaac\cobotta_pro_900\cobotta_pro_900.urdf",
                                                        end_effector_frame_name="onrobot_rg6_base_link",
                                                        maximum_substep_size=0.00334)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )

        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )