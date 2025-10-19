import torch
import numpy as np
import time





def quat_tensor_to_rot_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor of quaternion data into rotation matrices.
    
    The function assumes that each quaternion is in the form:
        [q_w, q_x, q_y, q_z]
    The expected input shape is either (n, 4) for a batch of quaternions or (4,) for a single quaternion.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (n, 4) or (4,).
 
    Returns:
        torch.Tensor: A tensor of shape (n, 3, 3) containing the corresponding rotation matrices.
    """
    # If a single quaternion is provided (1D tensor), add a batch dimension.
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # Now tensor has shape (1, 4)

    # Normalize the quaternion (adding a small epsilon to avoid division by zero)
    q_norm = torch.norm(tensor, dim=1, keepdim=True)
    q_normalized = tensor / (q_norm + 1e-8)

    # Extract quaternion components
    w = q_normalized[:, 0]
    x = q_normalized[:, 1]
    y = q_normalized[:, 2]
    z = q_normalized[:, 3]

    # Compute the rotation matrix elements using the standard conversion formula.
    # For a quaternion (w, x, y, z) the rotation matrix is:
    #
    #   R = [ [1 - 2*(y² + z²),   2*(x*y - w*z),   2*(x*z + w*y)],
    #         [2*(x*y + w*z),     1 - 2*(x² + z²), 2*(y*z - w*x)],
    #         [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x² + y²)] ]
    #
    R11 = 1 - 2 * (y * y + z * z)
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (x * z + w * y)

    R21 = 2 * (x * y + w * z)
    R22 = 1 - 2 * (x * x + z * z)
    R23 = 2 * (y * z - w * x)

    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)
    R33 = 1 - 2 * (x * x + y * y)

    # Stack the computed components into rotation matrices.
    # Each of R11, R12, ..., R33 has shape (n,), so we first create rows of shape (n, 3)
    row1 = torch.stack([R11, R12, R13], dim=1)
    row2 = torch.stack([R21, R22, R23], dim=1)
    row3 = torch.stack([R31, R32, R33], dim=1)
    
    # Stack the rows to get a final shape of (n, 3, 3)
    rot_matrices = torch.stack([row1, row2, row3], dim=1)
    
    return rot_matrices


def quat_tensor_to_rot_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor of quaternion data into rotation matrices.
    
    The function assumes that each quaternion is in the form:
        [q_w, q_x, q_y, q_z]
    The expected input shape is either (n, 4) for a batch of quaternions or (4,) for a single quaternion.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (n, 4) or (4,).
 
    Returns:
        torch.Tensor: A tensor of shape (n, 3, 3) containing the corresponding rotation matrices.
    """
    # If a single quaternion is provided (1D tensor), add a batch dimension.
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # Now tensor has shape (1, 4)

    # Normalize the quaternion (adding a small epsilon to avoid division by zero)
    q_norm = torch.norm(tensor, dim=1, keepdim=True)
    q_normalized = tensor / (q_norm + 1e-8)

    # Extract quaternion components
    w = q_normalized[:, 0]
    x = q_normalized[:, 1]
    y = q_normalized[:, 2]
    z = q_normalized[:, 3]

    # Compute the rotation matrix elements using the standard conversion formula.
    # For a quaternion (w, x, y, z) the rotation matrix is:
    #
    #   R = [ [1 - 2*(y² + z²),   2*(x*y - w*z),   2*(x*z + w*y)],
    #         [2*(x*y + w*z),     1 - 2*(x² + z²), 2*(y*z - w*x)],
    #         [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x² + y²)] ]
    #
    R11 = 1 - 2 * (y * y + z * z)
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (x * z + w * y)

    R21 = 2 * (x * y + w * z)
    R22 = 1 - 2 * (x * x + z * z)
    R23 = 2 * (y * z - w * x)

    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)
    R33 = 1 - 2 * (x * x + y * y)

    # Stack the computed components into rotation matrices.
    # Each of R11, R12, ..., R33 has shape (n,), so we first create rows of shape (n, 3)
    row1 = torch.stack([R11, R12, R13], dim=1)
    row2 = torch.stack([R21, R22, R23], dim=1)
    row3 = torch.stack([R31, R32, R33], dim=1)
    
    # Stack the rows to get a final shape of (n, 3, 3)
    rot_matrices = torch.stack([row1, row2, row3], dim=1)
    
    return rot_matrices.detach().cpu().numpy()


def quat_np_to_screw_np(q: np.ndarray) -> np.ndarray:
    """
    Converts a unit quaternion (numpy array of shape (4,))
    in the form [w, x, y, z] to its corresponding so(3) 3D rotation vector.
    
    The so(3) vector is given by theta * axis, where
      - theta = 2 * arccos(w) is the rotation angle,
      - axis = [x, y, z] / sin(theta/2) is the rotation axis.
    
    If sin(theta/2) is very small, a small-angle approximation is used.
    
    Parameters:
        q (np.ndarray): Input quaternion as a numpy array of shape (4,).
    
    Returns:
        np.ndarray: The corresponding so(3) rotation vector (3-dimensional).
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Input quaternion must be a 4-dimensional vector.")
    
    # Normalize the quaternion to ensure it's a unit quaternion.
    norm_q = np.linalg.norm(q)
    if norm_q < np.finfo(float).eps:
        raise ValueError("Zero quaternion provided; cannot convert.")
    q = q / norm_q

    w, x, y, z = q
    # Compute the rotation angle.
    theta = 2 * np.arccos(w)
    # Compute the sine of half the rotation angle.
    sin_half_theta = np.sqrt(1 - w * w)
    
    # If sin_half_theta is small, use a first order approximation.
    if sin_half_theta < 1e-8:
        # For very small angles, theta ~ 2*sin_half_theta and axis ~ [x, y, z]/sin_half_theta.
        # The rotation vector approximates to 2*[x, y, z].
        return np.array([2*x, 2*y, 2*z])
    else:
        # Otherwise, compute the axis properly.
        axis = np.array([x, y, z]) / sin_half_theta
        return theta * axis


import numpy as np

def posquat_to_se3(goals_quat: np.ndarray) -> np.ndarray:
    """
    Converts an array of goals, where each row is
      [pos_x, pos_y, pos_z, q_w, q_x, q_y, q_z],
    to an array where each row is
      [pos_x, pos_y, pos_z, so3_x, so3_y, so3_z].

    Also supports the one-row case by accepting a 1D array of length 7.
    
    Parameters:
        goals_quat (np.ndarray): Input array of shape (N, 7) or (7,).
        
    Returns:
        np.ndarray: Converted array of shape (N, 6) or (6,) if one row input.
    """
    input_was_1d = False
    # If the input is 1D, reshape it to 2D for uniform processing.
    if goals_quat.ndim == 1:
        if goals_quat.size != 7:
            raise ValueError("Input must have 7 elements if 1D.")
        goals_quat = goals_quat.reshape(1, -1)
        input_was_1d = True

    if goals_quat.shape[1] != 7:
        raise ValueError("Each row of the input array must have 7 elements: 3 for position and 4 for quaternion.")
    
    converted_goals = []
    for row in goals_quat:
        pos = row[:3]        # Extract position.
        quat = row[3:]       # Extract quaternion (assumed [w, x, y, z]).
        so3 = quat_np_to_screw_np(quat)  # Convert quaternion to so(3). Make sure this function is defined.
        # Concatenate the position with the so(3) vector.
        converted_goals.append(np.concatenate([pos, so3]))
    
    result = np.array(converted_goals)
    # If input was 1D, return a 1D array.
    if input_was_1d:
        result = result[0]
    return result


def se3_to_posquat(goals_se3: np.ndarray) -> np.ndarray:
    """
    Converts an array of goals, where each row is
      [pos_x, pos_y, pos_z, so3_x, so3_y, so3_z],
    to an array where each row is
      [pos_x, pos_y, pos_z, q_w, q_x, q_y, q_z].

    Also supports the one-row case by accepting a 1D array of length 6.
    
    The so(3) vector is interpreted as the rotation vector (axis-angle),
    where its norm is the rotation angle and its direction is the axis.
    
    Parameters:
        goals_so3 (np.ndarray): Input array of shape (N, 6) or (6,).
        
    Returns:
        np.ndarray: Converted array of shape (N, 7) or (7,) if one row input.
    """
    input_was_1d = False
    # Reshape 1D input to 2D for processing.
    if goals_se3.ndim == 1:
        if goals_se3.size != 6:
            raise ValueError("Input must have 6 elements if 1D.")
        goals_se3 = goals_se3.reshape(1, -1)
        input_was_1d = True

    if goals_se3.shape[1] != 6:
        raise ValueError("Each row of the input array must have 6 elements: 3 for position and 3 for so(3) vector.")
    
    converted_goals = []
    for row in goals_se3:
        pos = row[:3]  # Extract position.
        so3 = row[3:]  # Extract so(3) vector.
        
        # Compute the rotation angle.
        theta = np.linalg.norm(so3)
        
        # If the rotation angle is very small, return the identity quaternion.
        if theta < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = so3 / theta
            half_theta = theta / 2.0
            quat = np.concatenate(([np.cos(half_theta)], np.sin(half_theta) * axis))
        
        # Concatenate the position with the quaternion.
        converted_goals.append(np.concatenate([pos, quat]))
    
    result = np.array(converted_goals)
    # Return 1D array if input was 1D.
    if input_was_1d:
        result = result[0]
    return result


if __name__ == '__main__':
    start_time = time.perf_counter()
    # Example 1: Using a batch of quaternions (shape: (n, 4))
    ee_pose_w = torch.tensor([
        [1.0000e+00,  5.7377e-17,  2.2724e-08, -2.2724e-08],
        [1.0000e+00,  5.7377e-17,  2.2724e-08, -2.2724e-08]
    ], device='cuda:0') # on the device GPU
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for forward kinematic calculation: {elapsed_time:.6f} seconds")
    R_tensor = quat_tensor_to_rot_tensor(ee_pose_w)

    start_time = time.perf_counter()
    R_np = quat_tensor_to_rot_np(ee_pose_w)
    # Stop the high-resolution timer
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for forward kinematic calculation: {elapsed_time:.6f} seconds")
    print("Rotation matrices from batch of quaternions:")
    print(R_tensor)

    # Move the tensor to cpu first and then convert by .numpy()
    R_numpy= R_tensor.detach().cpu().numpy()
    # First, .detach() creates a new tensor that shares memory but no longer tracks gradients, so it's detached from the computation graph. Next, .cpu() ensures the tensor is moved from GPU to CPU if needed. Finally, .numpy() converts the tensor into a NumPy array
    print("Rotation matrices converted back to numpy array:",R_np)
    print("Rotation matrices converted back to numpy array:",R_numpy)

    torch_tensor = torch.from_numpy(R_numpy)

    cuda0 = torch.device('cuda:0')
    torch_tensor_gpu = torch_tensor.to(cuda0)


    ee_goals_quat = np.array([
        [0.2, 0.3, 0.7, 0.707, 0,     0.707, 0],
        [0.2, -0.4, 0.6, 0.707, 0.707, 0.0,   0.0],
        [0.2, 0,    0.3, 0.0,   1.0,   0.0,   0.0],
    ])  # Each row: [position (3 values), quaternion (4 values)]

    ee_goals_so3 = posquat_to_se3(ee_goals_quat)
    print("Converted goals (position + so(3) vector):")
    print(ee_goals_so3)

    ee_goals_quat = se3_to_posquat(ee_goals_so3)
    print(ee_goals_quat)
    

