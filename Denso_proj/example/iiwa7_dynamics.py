from __future__ import division
import pinocchio as pin, math
import numpy as np
from matplotlib import pyplot as plt
import os

def print_formatted_nd_matrix(matrix):
    """
    Prints a 1D, 2D, or 3D NumPy array in a structured and readable format.

    Parameters:
    - matrix (numpy.ndarray): A 1D, 2D, or 3D NumPy array.

    Returns:
    - None (prints the matrix in a structured format).
    """
    import numpy as np

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim == 3:
        N, M, K = matrix.shape  # Get dimensions for a 3D array
        for i in range(N):
            print(f"Matrix {i+1} ({M}×{K}):")  # Label each M×K matrix
            for row in matrix[i]:
                print(" ".join(f"{val:10.7f}" for val in row))  # Format values in the row
            print()  # Add spacing between matrices

    elif matrix.ndim == 2:
        M, K = matrix.shape  # Get dimensions for a 2D matrix
        print(f"Matrix ({M}×{K}):")
        for row in matrix:
            print(" ".join(f"{val:10.7f}" for val in row))
        print()  # Add spacing

    elif matrix.ndim == 1:
        L = matrix.shape[0]
        print(f"Vector (length {L}):")
        # Print each element on a new line or in a single row; here we choose one per line
        for val in matrix:
            print(f"{val:10.7f}")
        print()

    else:
        raise ValueError("Matrix dimension not supported: only 1D, 2D, and 3D arrays are supported")


def print_joint_tree(model, joint_id=0, prefix=""):
    """
    Recursively prints the joint and link tree.
    
    Parameters:
    - model: The Pinocchio model.
    - joint_id: The current joint index.
    - prefix: String prefix for indentation.
    """
    parent = model.parents[joint_id] if joint_id != 0 else None
    print(prefix + f"Joint {joint_id} {model.names[joint_id]}: parent={parent}")
    # Recursively print all children of the current joint.
    for child_id in model.children[joint_id]:
        print_joint_tree(model, child_id, prefix + "  ")


def print_model_inertias(model):
    """
    Prints the inertial parameters for each joint in the Pinocchio model.
    
    Note:
      - In Pinocchio, the inertial data from the URDF is merged into the joint frame.
        As a result, the relative placement of the inertial frame with respect to the
        joint frame is the identity rotation and zero translation.
      - Inertial parameters (mass, lever) are stored in model.inertias.
    
    Parameters:
      model: A Pinocchio Model built from a URDF.
    """
    for i in range(model.njoints):
        # Retrieve joint name and inertia
        joint_name = model.names[i]
        inertia = model.inertias[i]
        mass = inertia.mass
        lever = inertia.lever  # lever = mass * center-of-mass offset
        
        # Compute the center-of-mass (CoM) offset.
        if mass > 0:
            com_offset = lever / mass
        else:
            com_offset = np.zeros(3)
        
        # Print joint and inertial information.
        print(f"Joint {i} ({joint_name}):")
        print(" Relative placement of the inertial frame w.r.t. joint frame:")
        print(" Inertial parameters:")
        print(f"   Mass = {mass:8.5f}")
        print(f"   Center of Mass offset = {com_offset}")
        print("   Inertia matrix:")
        # Assume inertia.matrix returns a 3x3 numpy array.
        inertia_matrix = inertia.matrix()
        # for row in inertia_matrix:
        #     formatted_row = " ".join(f"{val:8.5f}" for val in row)
        #     print("    ", formatted_row)
        # print("-" * 40)
        print_formatted_nd_matrix(inertia_matrix)

# Replace 'my_robot.urdf' with the name of your URDF file.
urdf_filename ='Denso_proj/iiwa7_description.urdf'

# Build the full path by joining the current working directory with the file name
urdf_path = os.path.join(os.getcwd(), urdf_filename)

# load the urdf file of your robot model
model = pin.buildModelFromUrdf(urdf_path)
print('model name: ' + model.name)
# Create data required by the algorithms
data = model.createData()
NQ = model.nq
NV = model.nv
 
print_joint_tree(model)
print_model_inertias(model)

# Print the frame names and attachments for each

# Print the frame names and attachments for each
# for f in model.frames:
#     print(f.name, 'attached to joint #', f.parentJoint)

# observe the ee frame (for panda, "panda_hand_joint":18), and inertia properities
# for i, n in enumerate(model.frames):
#     print(i, n)


# print the dimensions of the configuration vector representation and the velocity
print('Dimension of the configuration vector representation: ' + str(NQ))
print('Dimension of the velocity: ' + str(NV))
# calculate the total mass of the model, put it in data.mass[0] and return it
total_Mass = pin.computeTotalMass(model, data)
print('Total mass of the model: ', data.mass[0])

# Generating joint position, angular velocity and angular acceleration using quintic polynomials
# 1. Generating position q, initial: 0 deg, end: 60 deg, ouput:rad
def mypoly_p(t):
  p = np.poly1d([12*60*math.pi/180/(2*3**5),-30*60*math.pi/180/(2*3**4),20*60*math.pi/180/(2*3**3),0,0,0])
  return p(t)

q = np.zeros((61, 7))
for i in range(0, 61):
    for j in range(7):
        q[i, j] = mypoly_p(0 + 3 / 60 * i) # 61*7 matrix, each column represents a sequence of joint angles


# 2. Generating angular velocity qdot, initial: 0 rad/s, end: rad/s, ouput:rad/s
def mypoly_v(t):
  p = np.poly1d([5*12*60*math.pi/180/(2*3**5),-4*30*60*math.pi/180/(2*3**4),3*20*60*math.pi/180/(2*3**3),0,0])
  return p(t)

qdot = np.zeros((61, 7))
for i in range(0, 61):
    for j in range(7):
        qdot[i, j] = mypoly_v(0 + 3 / 60 * i) # 61*7 matrix, each column represents a sequence of joint velocity

# 3. Generating angular acceleration qddot, initial: 0 rad/s^2, end: rad/s^2, ouput:rad/s^2
def mypoly_a(t):
  p = np.poly1d([4*5*12*60*math.pi/180/(2*3**5),-3*4*30*60*math.pi/180/(2*3**4),2*3*20*60*math.pi/180/(2*3**3),0])
  return p(t)

qddot = np.zeros((61, 7))
for i in range(0, 61):
    for j in range(7):
        qddot[i, j] = mypoly_a(0 + 3 / 60 * i) # 61*7 matrix, each column represents a sequence of joint acceleration

# Calculates the torque of each joint, return 1*7 vector
Torque = np.zeros((61, 7))
for i in range(0,61):
    tau = pin.rnea(model,data,q[i],qdot[i],qddot[i])   # 1*7 vector
    Torque[i][:] = tau.T  # 61*7 vector
    # print('The ' + str(i) + 'th Torque is: ')
    # print_formatted_nd_matrix(tau.T)


# Computes the generalized gravity contribution G(q), stored in data.g
G_Torque = np.zeros((61, 7))
for i in range(0,61):
    G_Tau = pin.computeGeneralizedGravity(model,data,q[i]) # 1*7 vector
    G_Torque[i][:] = G_Tau  # 61*7 vector
    # print('The ' + str(i) + 'th G_Tau is: ')
    # print_formatted_nd_matrix(G_Tau)


# Computes the upper triangular part of the joint space inertia matrix M, stored in data.M
M_Matrix = np.zeros((61,7,7))
for i in range(0,61):
    M_Temp = pin.crba(model,data,q[i])
    M_Matrix[i,:,:] = M_Temp
    # print('The ' + str(i) + 'th M_Matrix is: ')
    # print_formatted_nd_matrix(M_Temp)

#Computes the Coriolis Matrix C
C_Matrix = np.zeros((61,7,7))
for i in range(0,61):
    C_Temp = pin.computeCoriolisMatrix(model,data,q[i], qdot[i])
    C_Matrix[i,:,:] = C_Temp
    # print('The ' + str(i) + 'th C_Matrix is: ')
    # print_formatted_nd_matrix(C_Temp)

# Verify the anti-symmetric property of dM/dt - 2* C, take the fifth sequence as example
M = pin.crba(model,data,q[5])
dt = 1e-8
q_plus = pin.integrate(model,q[5],qdot[5]*dt)
M_plus = pin.crba(model,data,q_plus)
print('The q increment is: ')
print_formatted_nd_matrix(q[5])

dM = (M_plus - M)/dt

C = pin.computeCoriolisMatrix(model,data,q[5],qdot[5])
print('The ' + str(5) + 'th C_Matrix is: ')
print_formatted_nd_matrix(C)

A = dM - 2*C
print('A is: ')
print_formatted_nd_matrix(A)

res = A + A.T
print('res is: ')
print_formatted_nd_matrix( res)
