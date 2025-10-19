import xml.etree.ElementTree as ET
import numpy as np
import json
from copy import deepcopy

# --- Helper Functions ---

def rpy_to_rot(rpy):
    """Convert roll, pitch, yaw (in radians) to a 3×3 rotation matrix."""
    roll, pitch, yaw = rpy
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll),  np.cos(roll)]])
    R_y = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]])
    return R_z.dot(R_y).dot(R_x)

def get_transform(origin_elem):
    """
    Given an XML element (typically an <origin> element) with optional "xyz" and "rpy" attributes,
    return the corresponding 4×4 homogeneous transformation matrix.
    """
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin_elem is not None:
        if 'xyz' in origin_elem.attrib:
            xyz = [float(val) for val in origin_elem.attrib['xyz'].split()]
        if 'rpy' in origin_elem.attrib:
            rpy = [float(val) for val in origin_elem.attrib['rpy'].split()]
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(rpy)
    T[:3, 3] = xyz
    return T

def inertia_to_matrix(inertia_elem):
    """Convert an <inertia> element’s attributes into a 3×3 NumPy matrix."""
    ixx = float(inertia_elem.attrib.get('ixx', 0))
    ixy = float(inertia_elem.attrib.get('ixy', 0))
    ixz = float(inertia_elem.attrib.get('ixz', 0))
    iyy = float(inertia_elem.attrib.get('iyy', 0))
    iyz = float(inertia_elem.attrib.get('iyz', 0))
    izz = float(inertia_elem.attrib.get('izz', 0))
    return np.array([[ixx, ixy, ixz],
                     [ixy, iyy, iyz],
                     [ixz, iyz, izz]])

def matrix_to_inertia_dict(M):
    """Convert a 3×3 inertia matrix into a dictionary of URDF inertia attributes."""
    return {
        "ixx": M[0, 0],
        "ixy": M[0, 1],
        "ixz": M[0, 2],
        "iyy": M[1, 1],
        "iyz": M[1, 2],
        "izz": M[2, 2]
    }

def combine_inertia(m1, com1, I1, m2, com2, I2):
    """
    Combine two bodies’ inertial properties.
    Given masses m1 and m2, centers of mass com1 and com2, and inertia matrices I1 and I2
    (each expressed about its own COM), compute the composite mass, composite COM, and
    composite inertia (about the composite COM) using the parallel–axis theorem.
    """
    m_total = m1 + m2
    if m_total == 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com_total = (m1 * com1 + m2 * com2) / m_total
    def shift(I, m, com):
        r = com - com_total
        return I + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    I_total = shift(I1, m1, com1) + shift(I2, m2, com2)
    return m_total, com_total, I_total

def parse_inertial(link):
    """
    Return (mass, center_of_mass, inertia_matrix) for the given link.
    If the link is missing an <inertial> element, assume zero values.
    """
    inertial = link.find('inertial')
    if inertial is None:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    mass_elem = inertial.find('mass')
    m = float(mass_elem.attrib.get('value', 0))
    origin_elem = inertial.find('origin')
    T = get_transform(origin_elem)
    com = T[:3, 3]
    inertia_elem = inertial.find('inertia')
    I = inertia_to_matrix(inertia_elem)
    return m, com, I

def update_inertial(link, m, com, I):
    """
    Create (or update) the <inertial> element for link with mass m, center-of-mass com,
    and inertia matrix I.
    """
    inertial = link.find('inertial')
    if inertial is None:
        inertial = ET.SubElement(link, 'inertial')
    mass_elem = inertial.find('mass')
    if mass_elem is None:
        mass_elem = ET.SubElement(inertial, 'mass')
    mass_elem.attrib['value'] = str(m)
    origin_elem = inertial.find('origin')
    if origin_elem is None:
        origin_elem = ET.SubElement(inertial, 'origin')
    # Store the COM (with zero rotation)
    origin_elem.attrib['xyz'] = f"{com[0]} {com[1]} {com[2]}"
    origin_elem.attrib['rpy'] = "0 0 0"
    inertia_elem = inertial.find('inertia')
    if inertia_elem is None:
        inertia_elem = ET.SubElement(inertial, 'inertia')
    for key, val in matrix_to_inertia_dict(I).items():
        inertia_elem.attrib[key] = str(val)

def rot_to_rpy(R):
    """
    Convert a rotation matrix R (3×3) to roll, pitch, yaw angles (in radians).
    This function uses a standard conversion.
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return (x, y, z)

# --- Existing Merging Function (merges and removes the fixed joint and its child) ---

def merge_urdf(input_urdf, output_urdf, fixed_joint_name, output_transforms_file):
    """
    Collapse a specified fixed joint (and its child link) in the URDF.
    
    This function:
      1. Merges the child's inertial data into the parent's inertial element.
      2. Re-parents any joints that originally had the child as parent.
      3. Merges the child's <visual> and <collision> elements (adjusting their origins).
      4. Removes the fixed joint and its child link from the URDF.
    
    The transformation matrix from the parent to the child is stored in a JSON file.
    """
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # Build a dictionary of links by name.
    links = {link.attrib['name']: link for link in root.findall('link')}

    # Locate the fixed joint by name.
    joint_to_merge = None
    for joint in root.findall('joint'):
        if joint.attrib.get('name') == fixed_joint_name:
            joint_to_merge = joint
            break

    if joint_to_merge is None:
        print(f"Fixed joint '{fixed_joint_name}' not found in the URDF.")
        return

    if joint_to_merge.attrib.get('type') != 'fixed':
        print(f"Joint '{fixed_joint_name}' is not of type 'fixed'.")
        return

    parent_name = joint_to_merge.find('parent').attrib['link']
    child_name  = joint_to_merge.find('child').attrib['link']

    # --- Record the Fixed Joint Transform ---
    T_joint = get_transform(joint_to_merge.find('origin'))
    merged_transforms = [{
        "joint": fixed_joint_name,
        "parent": parent_name,
        "child": child_name,
        "transform": T_joint.tolist()
    }]

    # --- Merge Inertial Data ---
    m_parent, com_parent, I_parent = parse_inertial(links[parent_name])
    m_child, com_child, I_child = parse_inertial(links[child_name])
    child_com_homog = np.append(com_child, 1)
    com_child_in_parent = T_joint.dot(child_com_homog)[:3]
    R_joint = T_joint[:3, :3]
    I_child_in_parent = R_joint.dot(I_child).dot(R_joint.T)
    m_new, com_new, I_new = combine_inertia(m_parent, com_parent, I_parent,
                                            m_child, com_child_in_parent, I_child_in_parent)
    update_inertial(links[parent_name], m_new, com_new, I_new)

    # --- Re-parent Joints That Had the Child as Parent ---
    for joint in root.findall('joint'):
        if joint.find('parent').attrib['link'] == child_name:
            joint.find('parent').attrib['link'] = parent_name
            origin_elem = joint.find('origin')
            if origin_elem is not None:
                T_old = get_transform(origin_elem)
                T_new = T_joint.dot(T_old)
                new_xyz = T_new[:3, 3]
                new_rpy = rot_to_rpy(T_new[:3, :3])
                origin_elem.attrib['xyz'] = f"{new_xyz[0]} {new_xyz[1]} {new_xyz[2]}"
                origin_elem.attrib['rpy'] = f"{new_rpy[0]} {new_rpy[1]} {new_rpy[2]}"

    # --- Merge <visual> and <collision> Tags ---
    for tag in ['visual', 'collision']:
        for elem in links[child_name].findall(tag):
            new_elem = deepcopy(elem)
            origin_elem = new_elem.find('origin')
            if origin_elem is not None:
                T_elem = get_transform(origin_elem)
            else:
                T_elem = np.eye(4)
                origin_elem = ET.Element('origin', {'xyz':"0 0 0", 'rpy':"0 0 0"})
                new_elem.insert(0, origin_elem)
            T_total = T_joint.dot(T_elem)
            new_xyz = T_total[:3, 3]
            new_rpy = rot_to_rpy(T_total[:3, :3])
            origin_elem.attrib['xyz'] = f"{new_xyz[0]} {new_xyz[1]} {new_xyz[2]}"
            origin_elem.attrib['rpy'] = f"{new_rpy[0]} {new_rpy[1]} {new_rpy[2]}"
            links[parent_name].append(new_elem)

    # --- Remove the Fixed Joint and Its Child Link ---
    root.remove(joint_to_merge)
    root.remove(links[child_name])

    # Write out the merged URDF.
    tree.write(output_urdf)

    # Save the merged transform info (only for this fixed joint) to a JSON file.
    with open(output_transforms_file, 'w') as f:
        json.dump(merged_transforms, f, indent=4)

# --- New Function ---

def merge_mass_prop_to_parent_link_of_a_fixed_joint_child_link(input_urdf, output_urdf, fixed_joint_name):
    """
    Merge the inertial (mass) properties and geometry of the child link of a given fixed joint into its parent.
    
    This function:
      1. Finds the fixed joint (by fixed_joint_name) and its associated parent and child links.
      2. Transforms the child's inertial properties into the parent's frame.
      3. Combines the parent's and child's inertial properties and updates the parent's <inertial> element.
      4. Updates the child's <inertial> element so that its mass, center-of-mass, and inertia matrix are zero.
      5. Moves the child's geometry (<visual> and <collision> elements) into the parent's link,
         adjusting their origins according to the fixed joint transform.
    
    The URDF is read from input_urdf and the updated URDF is written to output_urdf.
    """
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # Build a dictionary of links by name.
    links = {link.attrib['name']: link for link in root.findall('link')}

    # Locate the fixed joint by name.
    joint = None
    for j in root.findall('joint'):
        if j.attrib.get('name') == fixed_joint_name:
            joint = j
            break

    if joint is None:
        print(f"Fixed joint '{fixed_joint_name}' not found in the URDF.")
        return

    if joint.attrib.get('type') != 'fixed':
        print(f"Joint '{fixed_joint_name}' is not of type 'fixed'.")
        return

    parent_name = joint.find('parent').attrib['link']
    child_name = joint.find('child').attrib['link']

    if parent_name not in links:
        print(f"Parent link '{parent_name}' not found.")
        return
    if child_name not in links:
        print(f"Child link '{child_name}' not found.")
        return

    parent_link = links[parent_name]
    child_link  = links[child_name]

    # --- Get the Fixed Joint Transform ---
    T_joint = get_transform(joint.find('origin'))
    R_joint = T_joint[:3, :3]

    # --- Parse Parent and Child Inertial Data ---
    m_parent, com_parent, I_parent = parse_inertial(parent_link)
    m_child, com_child, I_child = parse_inertial(child_link)
    
    # Transform the child's center of mass and inertia into the parent's frame.
    child_com_homog = np.append(com_child, 1)
    com_child_in_parent = T_joint.dot(child_com_homog)[:3]
    I_child_in_parent = R_joint.dot(I_child).dot(R_joint.T)
    
    # --- Combine Inertial Data ---
    m_new, com_new, I_new = combine_inertia(m_parent, com_parent, I_parent,
                                            m_child, com_child_in_parent, I_child_in_parent)
    update_inertial(parent_link, m_new, com_new, I_new)
    
    # --- Update Child Link's Inertial Data to "Zero" ---
    inertial = child_link.find('inertial')
    if inertial is None:
        inertial = ET.SubElement(child_link, 'inertial')
    # Set mass to zero.
    mass_elem = inertial.find('mass')
    if mass_elem is None:
        mass_elem = ET.SubElement(inertial, 'mass')
    mass_elem.attrib['value'] = "0"
    # Set the origin (center of mass) to zeros.
    origin_elem = inertial.find('origin')
    if origin_elem is None:
        origin_elem = ET.SubElement(inertial, 'origin')
    origin_elem.attrib['xyz'] = "0 0 0"
    origin_elem.attrib['rpy'] = "0 0 0"
    # Set the inertia matrix to zeros.
    inertia_elem = inertial.find('inertia')
    if inertia_elem is None:
        inertia_elem = ET.SubElement(inertial, 'inertia')
    inertia_elem.attrib['ixx'] = "0"
    inertia_elem.attrib['ixy'] = "0"
    inertia_elem.attrib['ixz'] = "0"
    inertia_elem.attrib['iyy'] = "0"
    inertia_elem.attrib['iyz'] = "0"
    inertia_elem.attrib['izz'] = "0"

    # --- Move Geometry (Visual and Collision) from Child to Parent ---
    for tag in ['visual', 'collision']:
        # Collect all geometry elements from the child.
        elems = list(child_link.findall(tag))
        for elem in elems:
            # Get the element's origin transform (if any)
            origin_elem = elem.find('origin')
            if origin_elem is not None:
                T_elem = get_transform(origin_elem)
            else:
                T_elem = np.eye(4)
                # Create an origin element if it does not exist.
                origin_elem = ET.Element('origin', {'xyz': "0 0 0", 'rpy': "0 0 0"})
                elem.insert(0, origin_elem)
            # Compute the total transform: fixed joint transform followed by the element's own origin.
            T_total = T_joint.dot(T_elem)
            new_xyz = T_total[:3, 3]
            new_rpy = rot_to_rpy(T_total[:3, :3])
            origin_elem.attrib['xyz'] = f"{new_xyz[0]} {new_xyz[1]} {new_xyz[2]}"
            origin_elem.attrib['rpy'] = f"{new_rpy[0]} {new_rpy[1]} {new_rpy[2]}"
            # Append a copy of the geometry element to the parent link.
            parent_link.append(deepcopy(elem))
            # Remove the geometry element from the child link.
            child_link.remove(elem)

    # Write out the updated URDF.
    tree.write(output_urdf)
    print(f"Merged mass and geometry from child '{child_name}' to parent '{parent_name}' for fixed joint '{fixed_joint_name}'.")

# --- Main (Example Usage) ---


if __name__ == "__main__":
    # Replace these paths with your actual file paths.
    input_urdf = r'C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\panda_arm_hand.urdf'
    output_urdf = r'C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\panda_arm_hand_merged_joint8.urdf'
    output_transforms_file = r'C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\merged_joint8_transforms.json'
    
    # Specify the fixed joint to merge (by its name).
    fixed_joint_name = "panda_joint8"
    
    merge_urdf(input_urdf, output_urdf, fixed_joint_name, output_transforms_file)


    # Example fixed joint name to use with merge_mass_prop_to_parent_link_of_a_fixed_joint_child_link
    # (this only transfers the mass/inertial properties, leaving the structure intact).
    fixed_joint_name_mass_merge = "panda_hand_joint"
    output_mass_merge_urdf = r'C:\Users\13306\OneDrive\Coding_Proj\issaaclab140\HEcode\franka_description\robots\panda_arm_hand_merged.urdf'
    merge_mass_prop_to_parent_link_of_a_fixed_joint_child_link(output_urdf, output_mass_merge_urdf, fixed_joint_name_mass_merge)
