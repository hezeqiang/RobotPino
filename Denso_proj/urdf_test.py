import os
import numpy as np
import modern_robotics as MR
# Defining PI
PI = np.pi

home_config_disp=np.array([
    [0, 0, 0.1975],
    [0.03, 0, 0.1975],
    [0, -0.34, 0],
    [-0.02, -0.197, 0],
    [0, 0, 0.143],
    [0, -0.08, 0]
],dtype=np.float32)

home_config_rot=np.array([
    [0, 0, 0],
    [-PI/2, 0, 0],
    [0, 0, 0],
    [PI/2, 0, 0],
    [-PI/2, 0, 0],
    [PI/2, 0, 0]
],dtype=np.float32)
# Space frame
# [1.     0.     0.]
# [0.     1.     0.]
# [0.     0.     1.]
# after rotation   [-PI/2, 0, 0]
# each column is the xyz axes unit vector in Space frame
# [1.     0.     0.]
# [0.     0.     1.]
# [0.     -1.    0.]

# Obtaining the row numbers of the numpy array
joint_number = home_config_disp.shape[0]
# print(joint_number)

home_config_M= np.random.rand(4, joint_number*4).astype(np.float32)
home_config_Space_M= np.random.rand(4, joint_number*4).astype(np.float32)

#  Obtaining the home_config of each joint frame
for i in range(joint_number):
    # print(home_config_rot[i])
    # print(i)
    Mso=MR.VecToso3(home_config_rot[i])
    MRot=MR.MatrixExp3(Mso)
    home_config_M[:, i*4:(i+1)*4]=MR.RpToTrans(MRot,home_config_disp[i])
    # print(home_config_M[:, i*4:(i+1)*4])


# calculate each Screw of joint in the base frame
# The Screw_omega is the last column (z axis) of the Rot matrix of each joint
# The Screw_q is the last column homogeneous matrix of each joint
########################################################################

# calculate the homogeneous matrix of each joint in the base frame
# T_1 ->T_6 calculation (including home_M) of the homogeneous matrix
home_config_Space_M[:, 0:4]=home_config_M[:, 0:4]
for i in range(joint_number-1):
    home_config_Space_M[:, (i+1)*4:(i+2)*4]=home_config_Space_M[:, (i)*4:(i+1)*4]@home_config_M[:, (i+1)*4:(i+2)*4]
    # print(home_config_M_Space[:, (i)*4:(i+1)*4])

home_config_Space_M[home_config_Space_M < 1e-8] = 0
# Space Screw Vector  // Body Screw vector notation: A
home_Screw_space_omega_v= np.random.rand(6, joint_number).astype(np.float32)

#  obtain origin of each joint frame in the base frame
for i in range(joint_number):
    # print(home_config_M_Space[0:3, i*4-1])
    # The Screw_omega is the last column (z axis) of the Rot matrix of each joint
    # The Screw_q (origin of frame) is the last column homogeneous matrix of each joint
    # assuming that z axis if the rotation axis, i.e., the rotation axis is [0:3,-1] of the 4 by 4 SE(3) matrix in the global coordinate
    home_Screw_space_omega_v[:, i] = MR.ScrewToAxis(home_config_Space_M[0:3, (i+1)*4-1],home_config_Space_M[0:3, (i+1)*4-2],0)
    # print(Screw_space_omega_v[:, i])


# Check the data type of the array
# print(home_config_M_Space.dtype)

# MR.ForwardDynamics
Thetalist =np.array([0, PI/3, PI/4, 0, PI/2, 0],dtype=np.float32)
T_end_effector= MR.FKinSpace(home_config_Space_M[:,20:24], home_Screw_space_omega_v, Thetalist)
print(T_end_effector)
# [[-9.65925802e-01  0.00000000e+00 -2.58819137e-01  6.37334262e-01]
#  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  2.69699267e-08]
#  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00  2.69699267e-08]
#  [ 2.58819137e-01  0.00000000e+00 -9.65925802e-01  4.19045932e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]


# Jacobi matrix calculation
#  the Screw vector considering the prev theta angle

Js = np.random.rand(6, joint_number).astype(np.float32)
Js = MR.JacobianSpace(home_Screw_space_omega_v, Thetalist)
print(Js)




