import numpy as np
from tool_merge_all_fixed_joint import merge_all_urdf
from tool_merge_one_fixed_joint import merge_mass_prop_to_parent_link_of_a_fixed_joint_child_link
from tool_merge_one_fixed_joint import merge_urdf
from tool_transfer_principal_axes import principal_axes_urdf_file
import subprocess
import time


# This script merges the fixed joint and its child link to the parent link 
# if the end-effector link is connected by a fixed joint, attribute the mass and inertia of the end-effector link to the parent link
# the end-effector link just includes a origin frame in the end-effector joint frame

def main():

    # address the urdf file
    input_urdf = r'franka_description\robots\panda_arm_hand.urdf'

    merged_urdf = r'franka_description\robots\panda_arm_hand_merged.urdf'

    merged_principal_axes_urdf = r'franka_description\robots\panda_arm_hand_merged_principal_axes.urdf'

    output_transforms_file = r'franka_description\robots\merged_transforms.json'

    # Specify the fixed joint to merge (by its name).
    fixed_joint_name = "panda_joint8"
    
    merge_urdf(input_urdf, merged_urdf, fixed_joint_name, output_transforms_file)

    # Example fixed joint name to use with merge_mass_prop_to_parent_link_of_a_fixed_joint_child_link
    # (this only transfers the mass/inertial properties, leaving the structure intact).
    fixed_joint_name_mass_merge = "panda_hand_joint"
    merge_mass_urdf = r'franka_description\robots\panda_arm_hand_merged_mass.urdf'
    merge_mass_prop_to_parent_link_of_a_fixed_joint_child_link(merged_urdf, merge_mass_urdf, fixed_joint_name_mass_merge)

    
    time.sleep(1)  # Pauses the program for 1 second

    principal_axes_urdf_file(merge_mass_urdf, merged_principal_axes_urdf)


    # convert urdf to usd format
    command = [
        "python",
        r"c:\Users\13306\OneDrive\Coding_Proj\issaaclab140\source\standalone\tools\convert_urdf.py",
        r"franka_description\robots\panda_arm_hand_merged_mass.urdf",
        r"\Denso_proj\panda_arm_hand_merged_mass.usd",
        "--fix-base"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

if __name__ == "__main__":
    main()

