import os
import numpy as np
import modern_robotics as MR
# Defining PI
PI = np.pi


home_config_pos=np.array([
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