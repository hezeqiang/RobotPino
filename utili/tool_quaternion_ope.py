import torch

def quat_tensor_to_rot_matrix(tensor: torch.Tensor) -> torch.Tensor:
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



if __name__ == '__main__':
    # Example 1: Using a batch of quaternions (shape: (n, 4))
    ee_pose_w = torch.tensor([
        [1.0000e+00,  5.7377e-17,  2.2724e-08, -2.2724e-08],
        [1.0000e+00,  5.7377e-17,  2.2724e-08, -2.2724e-08]
    ], device='cuda:0') # on the device GPU
    R_tensor = quat_tensor_to_rot_matrix(ee_pose_w)
    print("Rotation matrices from batch of quaternions:")
    print(R_tensor)

    # Move the tensor to cpu first and then convert by .numpy()
    R_numpy= R_tensor.detach().cpu().numpy()
    # First, .detach() creates a new tensor that shares memory but no longer tracks gradients, so it's detached from the computation graph. Next, .cpu() ensures the tensor is moved from GPU to CPU if needed. Finally, .numpy() converts the tensor into a NumPy array
    print("Rotation matrices converted back to numpy array:",R_numpy)

    torch_tensor = torch.from_numpy(R_numpy)

    cuda0 = torch.device('cuda:0')
    torch_tensor_gpu = torch_tensor.to(cuda0)
