# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the r068 robots.

The following configurations are available:

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

R068_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"Denso_proj\r068.usd",

        activate_contact_sensors=False,
        
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),
    actuators={
        # group 1 with the same paras and PD controller
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        # group 2 with the same paras and PD controller
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-6]"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),

    },
    soft_joint_pos_limit_factor=1.0,
    # Fraction specifying the range of DOF position limits (parsed from the asset) to use. Defaults to 1.0.
)
"""Configuration of r068 robot."""


R068_CFG_HIGH_PD_CFG = R068_CFG.copy()
R068_CFG_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
R068_CFG_HIGH_PD_CFG.actuators["shoulder"].stiffness = 0
R068_CFG_HIGH_PD_CFG.actuators["shoulder"].damping = 0

"""Configuration of R068 robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
