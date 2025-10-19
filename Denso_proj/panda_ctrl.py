import numpy as np
import time
import modern_robotics as MR
import math
from Denso_proj.RobotModel import RobotModel as RM
PI = math.pi

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


# Start a high-resolution timer
start_time = time.perf_counter()
panda_robot.cal_paras()
# Stop the high-resolution timer
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time for build model: {elapsed_time:.6f} seconds")
# panda_robot_instance.tool.rev_joint_list

Thetalist0 = np.array([0, 0, 0, 0, 0, 0, 0]) #  7 joint angle list

# Start a high-resolution timer
start_time = time.perf_counter()

T1 = MR.FKinBody(panda_robot.M, panda_robot.Bscrewlist, Thetalist0)
# T2 = MR.FKinSpace(panda_robot_instance.tool.M, panda_robot_instance.tool.Sscrewlist, Thetalist)

# Stop the high-resolution timer
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Elapsed time for forward kinematic calculation: {elapsed_time:.6f} seconds")

print(T1,"\n")
# print(T2,"\n")
# end-effector joint representation, not the end link origin frame representation
# verified by the isaacsim
# [[ 7.07106781e-01  7.07106781e-01  0.00000000e+00  8.80000000e-02]
#  [ 7.07106781e-01 -7.07106781e-01 -9.79316628e-12 -7.14901138e-13]
#  [-6.92481428e-12  6.92481428e-12 -1.00000000e+00  9.26000000e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# [0.707, 0.707 ,  0.0, 0.088],
# [0.707, -0.707,  0.0,   0.0],
# [0.0  ,    0.0, -1.0, 0.926],
# [0.0  ,    0.0,  0.0,   1.0],

Thetalist = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]) #  7 joint angle list
Thetalist_guess = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])*0.5 #  7 joint angle list

T1 = MR.FKinBody(panda_robot.M, panda_robot.Bscrewlist, Thetalist)

# Start a high-resolution timer
start_time = time.perf_counter()

Thetalist_Fikin = MR.IKinSpace(panda_robot.Sscrewlist, 
                      panda_robot.M, 
                      T1, 
                      Thetalist_guess, 
                      0.01, 
                      0.001)

# Stop the high-resolution timer
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Elapsed time for inverse kinematic calculation: {elapsed_time:.6f} seconds")

print(Thetalist_Fikin,"\n")
panda_robot.print_formatted_nd_matrix(Thetalist_Fikin[0])


J0= MR.JacobianSpace(panda_robot.Sscrewlist, Thetalist0)
panda_robot.print_formatted_nd_matrix(J0)
#   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
#   0.00000   1.00000   0.00000  -1.00000   0.00000  -1.00000  -0.00000
#   1.00000   0.00000   1.00000   0.00000   1.00000   0.00000  -1.00000
#   0.00000  -0.33300  -0.00000   0.64900   0.00000   1.03300   0.00000
#   0.00000   0.00000   0.00000  -0.00000   0.00000   0.00000   0.08800
#   0.00000   0.00000   0.00000  -0.08250   0.00000   0.00000  -0.00000

panda_robot.init_pin_model()
# Start a high-resolution timer

start_time = time.perf_counter()
panda_robot.calc_Blist_pin()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time for calc_Blist_pin: {elapsed_time:.6f} seconds")