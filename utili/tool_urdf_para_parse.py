import os
import numpy as np
from Denso_proj.RobotModel import RobotModel as RM
import json
import pinocchio as pin
import modern_robotics as MR

def read_last_link_ee_transforms(json_file):
    """
    Read the JSON file containing merged transforms and return a dictionary
    mapping each joint name to its 4x4 transformation matrix (as a NumPy array).
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    transforms = {}
    for entry in data:
        joint_name = entry.get('joint', 'unknown_joint')
        # Convert the stored list-of-lists back into a NumPy array.
        T = np.array(entry['transform'])
        transforms[joint_name] = {
            'parent': entry.get('parent', ''),
            'child': entry.get('child', ''),
            'transform': T
        }
    return transforms

if __name__ == "__main__":
    
    panda_robot_urdf_file =r"C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\panda_arm_hand_merged_principal_axes.urdf"

    with open (panda_robot_urdf_file, "r") as file:
        panda_robot_urdf = file.read()
    # True M
    M = np.array(
        [
            [0.707 , 0.707 , 0.0, 0.088],
            [0.707 , -0.707 , 0.0, 0.0],
            [0.0, 0, -1.0, 0.926],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    
    config={
                "body_frame": "panda_hand",
                "space_frame": "panda_link0",
                "namespace": "panda",
            }

    panda_robot = RM(config)
    panda_robot.load_desc_from_file(panda_robot_urdf_file)
    # notice the body_frame is the end-effector joint frame,not link origin frame


    panda_robot.cal_paras()
    panda_robot.init_pin_model()
    panda_robot.calc_Slist_pin()
    panda_robot.calc_Blist_pin()


    panda_robot.print_formatted_nd_matrix(panda_robot.M)


    T = np.array([[1, 0,  0, 0],
                    [0, 0, -1, 0],
                    [0, 1,  0, 3],
                    [0, 0,  0, 1]])
    se3 = panda_robot.MatrixLog6(T)

    print(se3)
    # panda_robot.print_formatted_nd_matrix(se3)
    M_SE3= panda_robot.MatrixExp6(se3)

    panda_robot.print_formatted_nd_matrix(M_SE3)

    print("NOTE: each frame in fixed in the link Center of Mass, and we call it as link frame")
    print("NOTE: the joint {i} is the parent of the link {i} ")
    print("NOTE: if the last link prev joint frame is not the end-effector frame, define: last_link_to_ee_frame")
    print("NOTE: notice the end-effector body_frame is the end-effector prev joint frame, not link origin frame")


    # # If the end-effector frame is not the last link frame,
    # # obtain parameter: last_link_to_ee_frame
    # # Read merged joint transformation matrix as numpy
    # json_file = r'C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\merged_transforms.json'

    # merged_link_to_ee_transforms_data = read_last_link_ee_transforms(json_file)
    # number_of_fixed_joint = len(merged_link_to_ee_transforms_data)

    # # Merge all the joint transformations to obtain the merged transform matrix
    # merged_transform = np.eye(4)
    # for joint_name, joint_data in merged_link_to_ee_transforms_data.items():
    #     merged_link_to_ee_transform = joint_data["transform"] @ merged_transform

    # panda_robot.cal_paras(last_link_to_ee_frame = merged_link_to_ee_transform)
