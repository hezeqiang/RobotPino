import numpy as np
import xml.etree.ElementTree as ET
import math
import itertools

# Define the tolerance to use throughout (1e-6).
TOL = 1e-8

def rotationMatrixToRPY(R):
    """
    Convert a rotation matrix (assumed proper 3x3) into roll, pitch, yaw (radians)
    using the standard URDF/ROS (intrinsic rpy) convention.
    
    This function assumes R rotates vectors from the principal frame into the original frame.
    """
    # Here we use the following conventions:
    #   roll  = rotation about x
    #   pitch = rotation about y
    #   yaw   = rotation about z
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < TOL
    if not singular:
        roll = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else:
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0.0

    # If any angle is very close to zero, set it exactly to zero.
    roll  = 0.0 if abs(roll)  < TOL else roll
    pitch = 0.0 if abs(pitch) < TOL else pitch
    yaw   = 0.0 if abs(yaw)   < TOL else yaw
    return roll, pitch, yaw

def rotation_angle_from_identity(R):
    """
    Compute the rotation angle (in radians) of a rotation matrix R relative to identity.
    (Uses the formula: angle = arccos((trace(R)-1)/2)).
    """
    # Clamp the value to avoid numerical issues.
    val = (np.trace(R) - 1) / 2
    val = max(min(val, 1.0), -1.0)
    return math.acos(val)

def diagonalize_inertial_minimal(inertial_str):
    """
    Given an XML string representing an <inertial> element (with <origin>, <mass>, and <inertia>),
    diagonalize the inertia tensor and determine the rotation from the original frame to the 
    principal-axes frame. Out of all possible diagonalizing transformations, choose the one that 
    has the smallest rotation angle (i.e. is as close as possible to the original orientation).
    
    Returns a new XML string for the <inertial> element with:
      - the same center-of-mass,
      - the mass unchanged,
      - the inertia expressed as a diagonal matrix (the principal moments),
      - the origin's rpy attribute set to the minimal rotation.
    """
    # Parse the input XML string.
    root = ET.fromstring(inertial_str)
    
    # Get the original origin.
    origin_elem = root.find('origin')
    xyz = [float(val) for val in origin_elem.get('xyz').split()]
    # (We assume the original rpy is zero; otherwise one could compose rotations.)
    
    # Get the mass.
    mass_val = float(root.find('mass').get('value'))
    
    # Get the inertia components.
    inertia_elem = root.find('inertia')
    ixx = float(inertia_elem.get('ixx'))
    iyy = float(inertia_elem.get('iyy'))
    izz = float(inertia_elem.get('izz'))
    ixy = float(inertia_elem.get('ixy'))
    ixz = float(inertia_elem.get('ixz'))
    iyz = float(inertia_elem.get('iyz'))
    
    # Build the symmetric inertia matrix.
    I = np.array([[ixx, ixy, ixz],
                  [ixy, iyy, iyz],
                  [ixz, iyz, izz]])
    
    # Compute eigenvalues and eigenvectors.
    # Note: np.linalg.eigh returns eigenvalues in ascending order.
    eigenvals, eigenvecs = np.linalg.eigh(I)
    
    # We now want to choose an assignment (a permutation of eigenvectors and sign flips)
    # that yields a rotation matrix R (whose columns are the new axes expressed in the original frame)
    # such that the rotation from the original frame (identity) is as small as possible.
    best_R = None
    best_perm = None
    best_signs = None
    best_angle = None
    best_diag = None  # To store the diagonal inertia in the chosen order.
    
    # The original frame is assumed to be the standard basis: [1,0,0], [0,1,0], [0,0,1].
    # We'll try all permutations of the eigenvectors.
    for perm in itertools.permutations(range(3)):
        # Build a candidate rotation matrix from eigenvectors in the order of the permutation.
        R_candidate = np.column_stack([ eigenvecs[:, perm[0]],
                                        eigenvecs[:, perm[1]],
                                        eigenvecs[:, perm[2]] ])
        # The corresponding diagonal inertia will be:
        diag_candidate = np.array([eigenvals[perm[0]],
                                   eigenvals[perm[1]],
                                   eigenvals[perm[2]]])
        # Try all sign combinations (2^3 = 8 possibilities).
        for signs in itertools.product([1, -1], repeat=3):
            R_trial = R_candidate.copy()
            R_trial[:, 0] *= signs[0]
            R_trial[:, 1] *= signs[1]
            R_trial[:, 2] *= signs[2]
            # Enforce a right-handed coordinate system.
            if np.linalg.det(R_trial) < 0:
                # Flip the sign of the third column.
                R_trial[:, 2] *= -1
                diag_trial = diag_candidate.copy()
            else:
                diag_trial = diag_candidate.copy()
            # R_trial rotates vectors from the principal frame to the original frame.
            # Therefore, the rotation from the original frame to the principal frame is R_trial^T.
            R_desired = R_trial.T
            angle = rotation_angle_from_identity(R_desired)
            if best_angle is None or angle < best_angle:
                best_angle = angle
                best_R = R_trial
                best_perm = perm
                best_signs = signs
                best_diag = diag_trial.copy()
                
    # Now best_R is the rotation matrix (whose columns are the chosen principal axes expressed in the original frame)
    # that minimizes the rotation angle from the identity.
    # The rotation from the original frame to the principal frame is R_desired = best_R^T.
    R_desired = best_R.T
    roll, pitch, yaw = rotationMatrixToRPY(R_desired)
    
    # Format the computed angles and inertia values:
    def fmt(val):
        # If a value is very small, output exactly 0.0; otherwise format to six decimals.
        return f"{0.0:.6f}" if abs(val) < TOL else f"{val:.6f}"
    
    # Create a new XML element for the inertial data.
    new_inertial = ET.Element('inertial')
    new_origin = ET.SubElement(new_inertial, 'origin')
    new_origin.set('xyz', f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    new_origin.set('rpy', f"{fmt(roll)} {fmt(pitch)} {fmt(yaw)}")
    
    new_mass = ET.SubElement(new_inertial, 'mass')
    new_mass.set('value', f"{mass_val:.6f}")
    
    new_inertia = ET.SubElement(new_inertial, 'inertia')
    # The diagonal entries will be the eigenvalues in the order corresponding to our chosen assignment.
    # (best_diag is a 3-element array in the order: new x, new y, new z.)
    new_inertia.set('ixx', fmt(best_diag[0]))
    new_inertia.set('iyy', fmt(best_diag[1]))
    new_inertia.set('izz', fmt(best_diag[2]))
    new_inertia.set('ixy', "0")
    new_inertia.set('ixz', "0")
    new_inertia.set('iyz', "0")
    
    # Return the new inertial element as an XML string.
    return ET.tostring(new_inertial, encoding='unicode')

# --- Process the URDF file with multiple links ---
def principal_axes_urdf_file(input_urdf_path, output_urdf_path):
    # Parse the URDF file
    tree = ET.parse(input_urdf_path)
    root = tree.getroot()
    
    # Iterate over all links in the URDF
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is not None:
            # Convert the <inertial> element to a string
            inertial_str = ET.tostring(inertial, encoding='unicode')
            
            # Diagonalize the inertia with minimal rotation using our function.
            new_inertial_str = diagonalize_inertial_minimal(inertial_str)
            
            # Parse the returned string into an XML element
            new_inertial_elem = ET.fromstring(new_inertial_str)
            
            # Replace the old inertial element with the new one.
            link.remove(inertial)
            link.append(new_inertial_elem)
    
    # Write the updated URDF to a new file.
    tree.write(output_urdf_path, encoding='unicode')
    
    
if __name__ == '__main__':
    input_file = r'C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\panda_arm_hand_merged_joint8.urdf'
    output_file = r'C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\panda_arm_hand_merged_principal_axes.urdf'

    principal_axes_urdf_file(input_file, output_file)
    print(f"Processed URDF saved to: {output_file}")