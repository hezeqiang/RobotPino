import scipy
import numpy as np
from urdf_parser_py.urdf import URDF
from packaging import version
from scipy.spatial.transform import Rotation as R
import modern_robotics as MR
import pinocchio as pin
import tempfile
from pinocchio.robot_wrapper import RobotWrapper
import re
from typing import Optional
import logging
"""
This module contains a class that can be used to calculate matrices required
for using the ModernRobotics Library. https://github.com/NxRLab/ModernRobotics
"""

class RobotModel(object):
    def __init__(self, configs):
        self.ee_frame_name = configs["ee_frame"]
        self.root_frame_name = configs["root_frame"]
        self.robot_urdf_namespace = configs.get("namespace")
    
        
        print("ee frame link",self.ee_frame_name)
        print("root frame link",self.root_frame_name)
        print("robot namespace",self.robot_urdf_namespace)

        # Set up a logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # Create a file handler that logs even debug messages
        fh = logging.FileHandler(r'RobotPino\Denso_proj\RobotModel.log', mode='w')
        fh.setLevel(logging.DEBUG)
        # Create a formatter and set it for the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # Add the file handler to the logger
        self.logger.addHandler(fh)


    def load_desc_from_param(self):
        """
        Gets the URDF from the robot_description parameter on the ROS parameter
        server
        """
        key = "%s/robot_description" % self.robot_urdf_namespace
        try:
            self.robot_urdf = URDF.from_parameter_server(key=key)
        except KeyError:
            print(
                (
                    "Error: `%s` not found on the ROS parameter server. "
                    "Check that it is loaded and under the right namespace."
                )
                % key
            )
            exit(1)

    def load_desc_from_file(self, file_path, package_dirs):
        """
        Gets the URDF from the robot's URDF file
        """
        self.urdf_file = file_path
        self.package_dirs= package_dirs
        self.robot_urdf = URDF.from_xml_file(file_path)

    def load_desc_from_xml_string(self, xml_string):
        """
        Gets the URDF from an xml string containing the URDF
        """
        self.robot_urdf = URDF.from_xml_string(xml_string)

    def populate(self):
        """
        Populates relevant variables and lists
        """
        # Print all link names
        print("Links:")
        for link in self.robot_urdf.links:
            print(link.name)

        # Print all joint names
        print("\nJoints:")
        for joint in self.robot_urdf.joints:
            print(joint.name)
        print("\n")

        self.links = self.robot_urdf.links
        self.joints = self.robot_urdf.joints

        # init empty lists
        self.link_list = []
        self.joint_list = []
        self.rev_joint_list = []
        
        # joint frame
        self.M_joint_relative_trans_list = [] # home homogeneous matrix of each joint frame in the root frame SE(3), for kinematic chain only
        self.M_joint_space_trans_list = []
        # self.joint_pos_space_list = [] # world position of each joint frame
        # self.joint_rot_space_list = [] # world rotational matrix of each joint frame

        # link frame
        self.M_link_origin_root_frame_list = [] # home homogeneous matrix of each link origin frame (at CoM) in the root frame SE(3), for dynamics of the model only
        self.link_CoM_to_parent_joint_frame_list = [] # home homogeneous matrix of each link origin frame (at CoM) in the prev joint SE(3), for dynamics of the model only
        
        self.M_link_rot_ee_axis_list = [] # the rotation axis list in the link frame, upper part of ee screw in the link frame
        
        self.M_link_rot_space_axis_list = [] # the rotation axis list in the root frame, upper part of ee screw in the link frame

        # Inertia properties
        self.link_com_list= [] # link centor of mass in prev joint frame (current link frame)      
        self.link_inertia_rpy_list= [] # frame of the inertia matrix in current link frame at center of mass
        self.link_mass_list= [] #mass
        self.link_inertia_matrix_list= [] # inertia matrix in current link frame at center of mass

        # find ee link in urdf tree
        self.ee_frame = self._search(
            self.links, "%s" % (self.ee_frame_name), "name"
        )

        self.root_frame = self._search(
            self.links, "%s" % (self.root_frame_name), "name"
        )

        if self.ee_frame is None or self.root_frame is None:
            raise ValueError("ee Frame or root Frame are empty")
        
        # for attr in dir(self.robot_urdf):
        #     if not attr.startswith("__"):  # Skip magic methods
        #         print(f"{attr}: {getattr(self.robot_urdf, attr)}")

    def get_M(self):
        """
        Returns the homogeneous transform matrix

        @return M
        """
        return self.M

    def get_Slist(self, transposed=False):
        """
        Returns the list of joint screw axes in the world frame

        @return Slist
        """
        if transposed:
            return self.Sscrewlist
        else:
            return self.Sscrewlist.T

    def cal_paras(self, last_link_to_ee_frame = [[1, 0, 0 ,0],[0, 1, 0 , 0],[0, 0 ,1, 0],[0 ,0, 0 ,1]]):
        """
        Builds relevant lists and calculates the M and Slist matrices
        """
        self.populate()
        self.build_link_lists()
        self.last_link_to_ee_frame=np.array(last_link_to_ee_frame) #from the hand joint frame to the ee frame, default [[1, 0, 0 ,0],[0, 1, 0 , 0],[0, 0 ,1, 0],[0 ,0, 0 ,1]]

        # Convert lists to NumPy arrays
        self.M_joint_relative_trans_list = np.array(self.M_joint_relative_trans_list) # shape: (n, 4, 4) if there are n transforms
        self.M_joint_space_trans_list = np.array(self.M_joint_space_trans_list) # shape: (n, 4, 4) if there are n transforms

        self.M_link_origin_root_frame_list = np.array(self.M_link_origin_root_frame_list) # shape: (n, 4, 4) if there are n transforms
        self.link_CoM_to_parent_joint_frame_list = np.array(self.link_CoM_to_parent_joint_frame_list) # shape: (n, 4, 4) if there are n transforms

        self.M_link_rot_ee_axis_list = np.array(self.M_link_rot_ee_axis_list) # shape: (n, 3) in ee frame
        self.M_link_rot_space_axis_list = np.array(self.M_link_rot_space_axis_list) # shape: (n, 3) in root frame

        self.link_com_list = np.array(self.link_com_list) # shape depends on collected COMs
        self.link_inertia_rpy_list = np.array(self.link_inertia_rpy_list)
        self.link_mass_list = np.array(self.link_mass_list)
        self.link_inertia_matrix_list = np.array(self.link_inertia_matrix_list)


        self.calc_M() # home-config homogeneous matrix of end-effector SE(3), root frame

        self.calc_Alist() # home-config screw axis of the end-effector in current link (prev joint) frame
        self.calc_Slist() # home-config screw axis of the end-effector in root frame
        self.calc_Blist() # home-config screw axis of the end-effector

        self._print_robot_info()

    def calc_M(self):
        """
        Calculates the homogeneous matrix describing the pose of the end
            effector in SE(3)
        """
        M = np.eye(4, dtype="float64")
        self.M_homo = np.dot(self.last_link_to_ee_frame, self.M_joint_space_trans_list[-1]) 
        self.M = pin.SE3(self.M_homo)

    def calc_Alist(self):
        """
        Calculates the list of screw axes
        """
        num_axes = len(self.rev_joint_list)  # number of revolute axes defines Slist shape
        Ascrew = np.zeros((6, num_axes))
        for i in range(num_axes):
            if self.joint_list[i] in self.rev_joint_list:
                w = np.array(self.M_link_rot_ee_axis_list[i], dtype="float64")
                # print("the parent joint in the link origin frame: ", -self.link_CoM_to_parent_joint_frame_list[i, :3, 3])
                q = np.array(-self.link_CoM_to_parent_joint_frame_list[i, :3, 3], dtype="float64")
                v = np.cross(-w, q)
                Ascrew[:, i] = np.hstack((v,w))
        self.Ascrewlist = Ascrew # the ee screw of prev joint with respect to each link origin frame

    def calc_Slist(self):
        """
        Calculates the list of screw axes
        """
        num_axes = len(self.rev_joint_list) # number of revolute axes defines Slist shape
        Sscrew = np.zeros((6, num_axes))
        for i in range(num_axes):
            if self.joint_list[i] in self.rev_joint_list:
                w = np.array(self.M_link_rot_space_axis_list[i], dtype="float64")
                q = np.array(self.M_joint_space_trans_list[i, :3, 3], dtype="float64")
                v = np.cross(-w, q)
                Sscrew[:, i] = np.hstack((v,w))
        self.Sscrewlist = Sscrew

    def calc_Blist(self):
        """
        Calculates the list of screw axes in the end-effector frame from the root screw
        """
        num_axes = len(self.rev_joint_list) # number of revolute axes defines Slist shape
        self.M_inv = self.M.inverse()
        # print(self.M_inv.homogeneous)

        ee_AdT_inv = self.M_inv.toActionMatrix()
        Bscrew = np.zeros((6, num_axes))
        for i in range(num_axes):
            Bscrew[:,i]= ee_AdT_inv @ self.Sscrewlist[:,i] #np.dot(ee_AdT_inv, self.Sscrewlist[:,i])

        self.Bscrewlist = Bscrew

    def build_link_lists(self):
        """
        Builds the transformation and link lists

        Iterates from the specified end effector link to the base frame
        """
        # obtain the ee link and joint (joint connect the later link)
        current_link = self.ee_frame
        parent_joint = self._get_parent_joint(self.ee_frame)


        # iterate backwards through link/joint tree up to base
        while current_link is not None and (current_link.name != self.root_frame.name):

            print("Joint name: ",parent_joint.name)

            self.link_list.insert(0, current_link)
            self.joint_list.insert(0, parent_joint)

            # keep track of revolute joints and their axes
            if parent_joint.type == "revolute":
                self.rev_joint_list.insert(0, parent_joint)

            self.M_joint_relative_trans_list.insert(0, self._get_parent_transform(current_link))

            if self._get_parent_axis(current_link): # if not none
                self.M_link_rot_ee_axis_list.insert(0, self._get_parent_axis(current_link))

            else: # fixed joint default axis
                self.M_link_rot_ee_axis_list.insert(0, [0.0, 0.0, 1.0])

            if self._get_com(current_link): # if not none

                self.link_com_list.insert(0, self._get_com(current_link)) 
                self.link_inertia_rpy_list.insert(0, self._get_inertia_rpy(current_link))

                self.link_mass_list.insert(0, self._get_mass(current_link))
                self.link_inertia_matrix_list.insert(0, self._get_inertia_matrix(current_link)) # 3*3

            else:  # no mass and inertia link default data
                self.link_com_list.insert(0, [0.0, 0.0, 0.0]) 
                self.link_inertia_rpy_list.insert(0, [0.0, 0.0, 0.0])

                self.link_mass_list.insert(0, 0)
                self.link_inertia_matrix_list.insert(0, np.array([
                                                        [0, 0, 0],
                                                        [0, 0, 0],
                                                        [0, 0, 0]
                                                        ]))


            current_link = self._get_parent_link(current_link)
            parent_joint = self._get_parent_joint(current_link)          

        print("The length of the link is (excluding the base link) ",len(self.joint_list),"\n")

        # print("The M_tf_list is",self.M_joint_relative_trans_list)
        # print("The M_axis_list is",self.M_link_rot_ee_axis_list)
        # print("The link_com_list is",self.link_com_list)

        M_joint_space_trans = np.eye(4, dtype="float64")
        for i in range(len(self.joint_list)):
            # joint frame 
            M_joint_space_trans = np.dot(M_joint_space_trans, self.M_joint_relative_trans_list[i])
            self.M_joint_space_trans_list.append(M_joint_space_trans)
            
            # link frame
            self.M_link_rot_space_axis_list.append(np.dot(M_joint_space_trans[:3, :3], self.M_link_rot_ee_axis_list[i]))

            link_CoM_to_parent_joint_frame = self._get_link_CoM_transform(self.link_com_list[i], self.link_inertia_rpy_list[i])
            
            self.link_CoM_to_parent_joint_frame_list.append(link_CoM_to_parent_joint_frame)
            self.M_link_origin_root_frame_list.append(np.dot(M_joint_space_trans,link_CoM_to_parent_joint_frame))


        # print("self.link_list (object): ",self.link_list)
        # print("self.joint_list (object): ",self.joint_list)

    def _get_joint_translation(self, joint):
        """
        Returns the translation of the given joint

        @param joint - joint to find the translation of
        @return translation from parent link to child link
        """
        if joint.origin is None or joint.origin.xyz is None:
            return np.array([0, 0, 0], dtype="float64")
        return np.array(joint.origin.xyz)

    def _get_joint_rotation(self, joint):
        """
        Returns the rotation of the given joint

        @param joint - joint to find the rotation of
        @return rotation from parent link to child link
        """
        if joint.origin is None or joint.origin.rpy is None:
            if version.parse(scipy.__version__) >= version.parse("1.4.0"):
                return R.from_euler("xyz", [0, 0, 0]).as_matrix()
            else:
                return R.from_euler("xyz", [0, 0, 0]).as_dcm()
        
        rpy = joint.origin.rpy
        if version.parse(scipy.__version__) >= version.parse("1.4.0"):
            return R.from_euler("xyz", rpy).as_matrix()
        else:
            return R.from_euler("xyz", rpy).as_dcm()

    def _get_parent_transform(self, link):
        """
        Returns the transformation from parent to child given the child link

        @param link - child link of joint
        @return transformation from parent to child
        """
        joint = self._get_parent_joint(link)
        T = np.eye(4, dtype="float64")
        if joint is None:
            return T
        else:
            T[:3, :3] = self._get_joint_rotation(joint)
            T[:3, 3] = self._get_joint_translation(joint)
            # print("transformation from parent link frame to child link frame (Prev joint frame of each link, not the CoM frame): \n",T)
            return T

    def _get_link_CoM_transform(self, pos_CoM, rpy_CoM):
        """
        Returns the transformation from parent joint to center of Mass of the child link

        @param link - child link of joint
        @return transformation from parent joint to child link
        self.link_inertia_rpy_list,  self.link_com_list
        """

        T = np.eye(4, dtype="float64")

        if version.parse(scipy.__version__) >= version.parse("1.4.0"):
            T[:3, :3] = R.from_euler("xyz", rpy_CoM).as_matrix()
        else:
            T[:3, :3] =  R.from_euler("xyz", rpy_CoM).as_dcm()
        
        T[:3, 3] = np.array(pos_CoM)
        # print("transformation from parent link frame to child link frame (Prev joint frame of each link, not the CoM frame): \n",T)
        return T

    def _get_parent_link(self, link):
        """
        Returns the parent link given its child

        @param link - child link to find the parent of
        @return parent link to given link
        @note returns None if there is no parent link
        """
        parent_joint = self._get_parent_joint(link)
        if parent_joint is None:
            return None
        return self._search(self.links, parent_joint.parent, "name")

    def _get_parent_joint(self, link):
        """
        Returns the parent joint given its child link

        @param link - child link to find the parent joint of
        @return parent joint to given link
        """
        joint = self._search(self.joints, link.name, "child")
        return joint

    def _get_parent_axis(self, link):
        """
        Returns the parent joint axis given its child

        @param link - child link to the find the parent joint's axis of
        @return axis about which the parent joint rotates
        """
        if self._get_parent_joint(link).type != "revolute":
            return None
        else:
            axis = self._get_parent_joint(link).axis
            return axis

    def _search(self, list_, key, attr):
        """
        Helper function to perform a key search through a list of objects

        @param list_ - list of objects to search through
        @param key - term to search for
        @param attr - attribute to look for key in for each object
        @return the first matching object that has an attribute matching the key
        @note returns None if no object attribute matches the key
        """
        result = [element for element in list_ if getattr(element, attr) == key]
        if len(result) > 0:
            return result[0]
        else:
            return None

    def _get_com(self, link):
        """
        Returns the center of mass (COM) of a given link.

        Parameters:
        - link (urdf_parser_py.urdf.Link): The link whose COM is needed.

        Returns:
        - list[float]: A list [x, y, z] representing the COM coordinates.
        - None: If the link has no inertial data.
        """
        return link.inertial.origin.xyz if link.inertial else None  # Center of mass

    def _get_inertia_rpy(self, link):
        """
        Returns the roll-pitch-yaw (RPY) orientation of the inertia frame
        relative to the link frame.

        Parameters:
        - link (urdf_parser_py.urdf.Link): The link whose inertia RPY is needed.

        Returns:
        - list[float]: A list [roll, pitch, yaw] representing the orientation.
        - None: If the link has no inertial data.
        """
        return link.inertial.origin.rpy if link.inertial else None  # Roll-Pitch-Yaw

    def _get_mass(self, link):
        """
        Returns the mass of a given link.

        Parameters:
        - link (urdf_parser_py.urdf.Link): The link whose mass is needed.

        Returns:
        - float: The mass of the link in kg.
        - None: If the link has no inertial data.
        """
        return link.inertial.mass if link.inertial else None  # Mass in kg

    def _get_inertia_matrix(self, link):
        """
        Returns the 3x3 inertia matrix for a given link.

        Parameters:
        - link (urdf_parser_py.urdf.Link): The link whose inertia matrix is needed.

        Returns:
        - np.ndarray: A 3x3 inertia matrix as a NumPy array.
        - None: If the link has no inertia data.
        """
        if not link.inertial:  # Check if inertial data exists
            return None  # No inertia information

        inertia = link.inertial.inertia  # Get inertia object

        # Construct the 3x3 inertia matrix
        inertia_matrix = np.array([
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz]
        ])

        return inertia_matrix
    
    def _print_robot_info(self):

        print("self.M_joint_relative_trans_list (joint transformation matrix of SE(3) expressed in the parent joint frame): \n")
        self.print_formatted_nd_matrix(self.M_joint_relative_trans_list)

        print("self.M_joint_space_trans_list (joint transformation matrix of SE(3) expressed in the root frame): \n")
        self.print_formatted_nd_matrix(self.M_joint_space_trans_list)
              
        print("self.M_link_origin_root_frame_list (each link origin frame (at CoM) transformation matrix of SE(3) expressed in the root frame): \n")
        self.print_formatted_nd_matrix(self.M_link_origin_root_frame_list)

        print("self.link_CoM_to_parent_joint_frame_list (each link origin frame (at CoM) transformation matrix of SE(3) expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_CoM_to_parent_joint_frame_list)

        print("self.M_link_rot_ee_axis_list (axis of each revolute joint expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.M_link_rot_ee_axis_list)
        
        print("self.M_link_rot_space_axis_list (axis of each revolute joint expressed in the root frame): \n")
        self.print_formatted_nd_matrix(self.M_link_rot_space_axis_list)

        print("self.link_com_list (CoM of each link expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_com_list)

        print("self.link_inertia_rpy_list (Roll-Pitch-Yaw of each link expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_inertia_rpy_list)

        print("self.link_mass_list (mass of each link expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_mass_list)

        print("self.link_inertia_matrix_list (Inertia matrix of each link expressed in the current link frame, i.e., current joint frame): \n")
        self.print_formatted_nd_matrix(self.link_inertia_matrix_list)

        print("self.Ascrewlist (root frame screw): \n")
        self.print_formatted_nd_matrix(self.Ascrewlist)

        print("self.Sscrewlist (link frame screw): \n")
        self.print_formatted_nd_matrix(self.Sscrewlist)

        print("self.Bscrewlist (root frame screw): \n")
        self.print_formatted_nd_matrix(self.Bscrewlist)

        print("self.M (home config homogeneous matrix of SE(3)): \n")
        self.print_formatted_nd_matrix(self.M.homogeneous)

    def print_formatted_nd_matrix(self, matrix):
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
                    print(" ".join(f"{val:9.5f}" for val in row))  # Format values in the row
                print()  # Add spacing between matrices

        elif matrix.ndim == 2:
            M, K = matrix.shape  # Get dimensions for a 2D matrix
            print(f"Matrix ({M}×{K}):")
            for row in matrix:
                print(" ".join(f"{val:9.5f}" for val in row))
            print()  # Add spacing

        elif matrix.ndim == 1:
            L = matrix.shape[0]
            print(f"Vector (length {L}):")
            # Print each element on a new line or in a single row; here we choose one per line
            for val in matrix:
                print(f"{val:9.5f}")
            print()

        else:
            raise ValueError("Matrix dimension not supported: only 1D, 2D, and 3D arrays are supported")


    # Pinocchio lib based fonctions

    def init_pin_model(self):
        """
        Initializes the Pinocchio model and data using the current URDF robot description.
        The Pinocchio model is built from the URDF XML string.
        """
        # Ensure the URDF has been loaded
        if not hasattr(self, "robot_urdf"):
            raise ValueError("Robot URDF not loaded. Please load a URDF description first using load_desc_from_file, load_desc_from_param, or load_desc_from_xml_string.")
        
        # Convert the URDF object to an urdf file
        try:
            urdf_xml = self.robot_urdf.to_xml_string()
        except AttributeError:
            raise AttributeError("The URDF object does not have a 'to_xml_string' method. Ensure your URDF is in the correct format.")

        # Build the Pinocchio model and create its data object
        self.model = pin.buildModelFromUrdf(self.urdf_file)

        self.data = self.model.createData()

        # self.print_joint_tree(self.model)
        self.print_model_inertias(self.model)
        # print(self.model)

        print("Listing all frames in the model:")
        for frame in self.model.frames:
            frame_id = self.model.getFrameId(frame.name)
            # Each frame has a name and an id (among other properties)
            print(f"Frame Name: {frame.name}, Frame ID: {frame_id}, Parent Joint: {frame.parentJoint}")
            # print(frame)
    
        # Retrieve the frame ID for the given joint name.
        try:
            self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        except Exception as e:
            raise ValueError("Frame with name '{}' not found in the Pinocchio model.".format(self.ee_frame_id))

        # Retrieve the frame ID for the given joint name.
        try:
            self.root_frame_id = self.model.getFrameId(self.root_frame_name)
        except Exception as e:
            raise ValueError("Frame with name '{}' not found in the Pinocchio model.".format(self.root_frame_id))

        # Initialize the configuration vector. Using pin.neutral ensures the vector has the right size.

        self.controller = self.Controller(self.model, self.data)

        self.reset_joints_pin()

        # print("Pinocchio model and data have been successfully initialized.")

    def reset_joints_pin(self):
        # Initialize the configuration vector. Using pin.neutral ensures the vector has the right size.
        # "panda_joint1": 0.0,
        # "panda_joint2": -0.569,
        # "panda_joint3": 0.0,
        # "panda_joint4": -2.810,
        # "panda_joint5": 0.0,
        # "panda_joint6": 3.037,
        # "panda_joint7": 0.741,
        # "panda_finger_joint.*": 0.04,
        default_q = np.array([0, -0.569, 0, -2.810, 0, 3.037, 0.741, 0.04, 0.04]) 
        # default_q = np.array([0, 0, 0, 0, 0, 3.037, 0, 0, 0]) 
        # Compute forward kinematics and update joint Jacobians.
        pin.forwardKinematics(self.model, self.data, default_q)
        pin.updateFramePlacements(self.model, self.data)

        # Retrieve the placement (as a pin.SE3 object) of the specified frame.
        default_M_pin = self.data.oMf[self.ee_frame_id]

        # the home homogeneous transformation matrix (4×4 numpy array).
        self.default_M = default_M_pin.homogeneous
        print("Default joint transformation matrix: ")
        self.print_formatted_nd_matrix(self.default_M)

        """
        # q = pin.neutral(self.model)

        # Create a dictionary for the main joints.
        # default_angles = {
        #     "panda_joint1": 0.0,
        #     "panda_joint2": -0.569,
        #     "panda_joint3": 0.0,
        #     "panda_joint4": -2.810,
        #     "panda_joint5": 0.0,
        #     "panda_joint6": 3.037,
        #     "panda_joint7": 0.741,
        # }

        # # Assign each main joint its default value.
        # for joint_name, angle in default_angles.items():
        #     # Get the joint id. Note: in Pinocchio, the first joint (the universe joint) has no dof.
        #     joint_id = self.model.getJointId(joint_name)
        #     # Each joint stores the index where its variables start in q.
        #     idx = self.model.joints[joint_id].idx_q
        #     q[idx] = angle  # Works if the joint has 1 degree-of-freedom

        # # For the finger joints that match the pattern "panda_finger_joint.*",
        # # iterate over all joints in the model and update the ones that match.
        # finger_pattern = re.compile(r"panda_finger_joint")
        # for j in range(len(self.model.names)):  # model.names is a list of joint names.
        #     if finger_pattern.match(self.model.names[j]):
        #         idx = self.model.joints[j].idx_q
        #         q[idx] = 0.04
        """

    def print_joint_tree(self, model, joint_id=0, prefix=""):
        """
        Recursively prints the joint and link tree.
        
        Parameters:
        - model: The Pinocchio model.
        - joint_id: The current joint index.
        - prefix: String prefix for indentation.
        """
    # Determine the parent index (for the root, show None)
        parent = model.parents[joint_id] if joint_id != 0 else None
        print(prefix + f"Joint {joint_id} {model.names[joint_id]}: parent={parent}")

        # Recursively print all children of the current joint.
        for child_id in model.children[joint_id]:
            self.print_joint_tree(model, child_id, prefix + "  ")

    def print_model_inertias(self,model):
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


            # the off-diagonal Non-zero value reflects the physical property of the link — the offset of its center of mass relative to its joint frame.
            for row in inertia_matrix:
                formatted_row = " ".join(f"{val:8.5f}" for val in row)
                print("    ", formatted_row)
            print("-" * 40)

    def get_joint_path_pin(self):
        """
        Computes the joint path (chain) from the root frame to the ee (end-effector) frame.
        Returns:
            list[int]: A list of joint IDs (excluding the base fixed joint) along the kinematic chain.
        """
        # Get the frame IDs for the root and ee frames.
        base_frame_id = self.model.getFrameId(self.root_frame_name)
        ee_frame_id   = self.model.getFrameId(self.ee_frame_name)

        # Each frame in Pinocchio is attached to a joint. Retrieve the parent joint IDs.
        base_joint_id = self.model.frames[base_frame_id].parent
        ee_joint_id   = self.model.frames[ee_frame_id].parent

        # Build the chain by traversing from the end-effector joint to the base joint.
        chain = []
        j = ee_joint_id
        # Traverse until we reach the base joint.
        while j != base_joint_id:
            chain.append(j)
            j = self.model.parents[j]
        # Reverse the chain so that it goes from base to end-effector.
        chain.reverse()
        return chain

# utility functions from Modern robotics
    @staticmethod
    def NearZero(z):
        """Determines whether a scalar is small enough to be treated as zero

        :param z: A scalar input to check
        :return: True if z is close to zero, false otherwise

        Example Input:
            z = -1e-7
        Output:
            True
        """
        return abs(z) < 1e-5

    @staticmethod
    def Normalize(V):
        """Normalizes a vector

        :param V: A vector
        :return: A unit vector pointing in the same direction as V

        Example Input:
            V = np.array([1, 2, 3])
        Output:
            np.array([0.26726124, 0.53452248, 0.80178373])
        """
        return V / np.linalg.norm(V)

    @staticmethod
    def RotInv(R):
        """Inverts a rotation matrix

        :param R: A rotation matrix
        :return: The inverse of R

        Example Input:
            R = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
        Output:
            np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0]])
        """
        return np.array(R).T

    @staticmethod
    def so3Tomatrix(omg):
        """Converts a 3-vector to an so(3) representation

        :param omg: A 3-vector
        :return: The skew symmetric representation of omg

        Example Input:
            omg = np.array([1, 2, 3])
        Output:
            np.array([[ 0, -3,  2],
                      [ 3,  0, -1],
                      [-2,  1,  0]])
        """
        return np.array([[0, -omg[2], omg[1]],
                         [omg[2], 0, -omg[0]],
                         [-omg[1], omg[0], 0]])

    @staticmethod
    def matrixToso3(so3mat):
        """Converts an so(3) representation to a 3-vector

        :param so3mat: A 3x3 skew-symmetric matrix
        :return: The 3-vector corresponding to so3mat

        Example Input:
            so3mat = np.array([[ 0, -3,  2],
                               [ 3,  0, -1],
                               [-2,  1,  0]])
        Output:
            np.array([1, 2, 3])
        """
        return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

    @staticmethod
    def AxisAng3(expc3):
        """Converts a 3-vector of exponential coordinates for rotation into
        axis-angle form

        :param expc3: A 3-vector of exponential coordinates for rotation
        :return: (unit rotation axis, rotation angle)

        Example Input:
            expc3 = np.array([1, 2, 3])
        Output:
            (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
        """
        return (RobotModel.Normalize(expc3), np.linalg.norm(expc3))

    @staticmethod
    def matrixTose3(se3mat): # --> [v,w]
        """Converts an se3 matrix into a spatial velocity vector

        :param se3mat: A 4x4 matrix in se3
        :return: The spatial velocity 6-vector corresponding to se3mat

        Example Input:
            se3mat = np.array([[ 0, -3,  2, 4],
                               [ 3,  0, -1, 5],
                               [-2,  1,  0, 6],
                               [ 0,  0,  0, 0]])
        Output:
            np.array([1, 2, 3, 4, 5, 6])
        """
        return np.r_[ [se3mat[2][1], se3mat[0][2], se3mat[1][0]],[se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

    @staticmethod
    def ScrewToAxis(q, s, h): # -->[w,q]
        """Takes a parametric description of a screw axis and converts it to a
        normalized screw axis

        :param q: A point lying on the screw axis
        :param s: A unit vector in the direction of the screw axis
        :param h: The pitch of the screw axis
        :return: A normalized screw axis described by the inputs

        Example Input:
            q = np.array([3, 0, 0])
            s = np.array([0, 0, 1])
            h = 2
        Output:
            np.array([0, 0, 1, 0, -3, 2])
        """
        return np.r_[np.cross(q, s) + h * s , s ]

    @staticmethod
    def AxisAng6(expc6): # -->[w,q]
        """Converts a 6-vector of exponential coordinates into screw axis-angle
        form

        :param expc6: A 6-vector of exponential coordinates for rigid-ee motion S*theta
        :return: (normalized screw axis, theta)

        Example Input:
            expc6 = np.array([1, 0, 0, 1, 2, 3])
        Output:
            (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
        """
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
        if RobotModel.NearZero(theta):
            theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
        return (np.array(expc6) / theta, theta)

    @staticmethod
    def so3ToSO3(so3):

        # Convert the rotation vector (in so(3)) to a rotation matrix using the exponential map.
        # Note: pin.SO3.Exp returns an SO3 object, and we take its matrix() for a 3x3 rotation matrix.
        R = pin.exp3(so3)
        return  R

    @staticmethod
    def se3_to_SE3(se3:np.ndarray)-> pin.SE3: 
        """
        Convert a 6D vector (translation + rotation vector) or an array of 6D vectors 
        into pinocchio.SE3 objects.
        
        Parameters:
            vec: np.array of shape (6,) or (N,6)
                - If shape is (6,), then vec[:3] is the translation and vec[3:] is the rotation vector.
                - If shape is (N,6), each row represents a separate SE3 pose.
        
        Returns:
            If input is a single 6D vector (shape (6,)):
                A pinocchio.SE3 object.
            If input is an array of 6D vectors (shape (N,6)):
                A list of pinocchio.SE3 objects.
        """
        # Check the dimensions of the input
        if se3.ndim == 1:
            # Single 6D vector case.
            translation = se3[:3]
            rot_vec = se3[3:]
            # Compute the rotation matrix from the rotation vector using the exponential map.
            # Note: pin.exp3 expects a 3D rotation vector and returns a 3x3 rotation matrix.
            R = pin.exp3(rot_vec)
            return pin.SE3(R, translation)
        
        else:
            raise ValueError("Input must be a 6D vector (shape (6,)) or an array of 6D vectors (shape (1,6)).")

    @staticmethod
    def se3_to_SE3_inverse(se3:np.ndarray)->pin.SE3:
        """
        Convert a 6D vector (translation + rotation vector) or an array of 6D vectors 
        into pinocchio.SE3 objects.
        
        Parameters:
            vec: np.array of shape (6,) or (N,6)
                - If shape is (6,), then vec[:3] is the translation and vec[3:] is the rotation vector.
                - If shape is (N,6), each row represents a separate SE3 pose.
        
        Returns:
            If input is a single 6D vector (shape (6,)):
                A pinocchio.SE3 object.
            If input is an array of 6D vectors (shape (N,6)):
                A list of pinocchio.SE3 objects.
        """
        # Check the dimensions of the input
        if se3.ndim == 1:
            # Single 6D vector case.
            translation = se3[:3]
            rot_vec = se3[3:]
            # Compute the rotation matrix from the rotation vector using the exponential map.
            # Note: pin.exp3 expects a 3D rotation vector and returns a 3x3 rotation matrix.
            R = pin.exp3(rot_vec)
            return pin.SE3(R, translation).inverse()
    
        
        else:
            raise ValueError("Input must be a 6D vector (shape (6,)) or an array of 6D vectors (shape (1,6)).")

    @staticmethod
    def SO3Toso3(so3):

        # Convert the rotation vector (in so(3)) to a rotation matrix using the exponential map.
        # Note: pin.SO3.Exp returns an SO3 object, and we take its matrix() for a 3x3 rotation matrix.
        R = pin.log3(so3)
        return 

    @staticmethod
    def SE3Tose3(SE3):
        return  pin.log6(SE3) # v,w

    @staticmethod
    def SE3ToQuat(SE3_obj:pin.SE3)->np.ndarray:
        """
        Convert a Pinocchio SE3 object to a 7D Pose Vector [x, y, z, qw, qx, qy, qz].

        Parameters:
            SE3_obj (pin.SE3): A Pinocchio SE3 object representing a transformation.

        Returns:
            np.ndarray: A 7D numpy array [x, y, z, qw, qx, qy, qz].
        """
        # Extract translation (position)
        position = SE3_obj.translation  # (x, y, z)

        # Convert rotation matrix to quaternion (qx, qy, qz, qw)
        quaternion = R.from_matrix(SE3_obj.rotation).as_quat()  # Default: (qx, qy, qz, qw)

        # Reorder to (qw, qx, qy, qz)
        quaternion = np.roll(quaternion, shift=1)

        # Concatenate translation and quaternion
        pose_vector = np.hstack((position, quaternion))

        return pose_vector


    @staticmethod
    def Adjoint(T): # motionaction
        """Computes the adjoint representation of a homogeneous transformation
        matrix

        :param T: A homogeneous transformation matrix
        :return: The 6x6 adjoint representation of T

        Example Input:
            T = np.array([[1, 0,  0, 0],
                          [0, 0, -1, 0],
                          [0, 1,  0, 3],
                          [0, 0,  0, 1]])
        Output:
            np.array([[1, 0,  0, 0, 0,  0],
                      [0, 0, -1, 0, 0,  0],
                      [0, 1,  0, 0, 0,  0],
                      [0, 0,  3, 1, 0,  0],
                      [3, 0,  0, 0, 0, -1],
                      [0, 0,  0, 0, 1,  0]])
        """
        T = pin.SE3(T)
        return T.toActionMatrix()

    def calc_M_pin(self):
        """
        Computes and returns home config SE(3) in the root (world) frame for a single joint.
        The joint is specified by its associated frame name. If no joint_name is provided,
        self.ee_frame_name is used.
        """

        # Ensure the Pinocchio model and data have been initialized.
        if not hasattr(self, 'model') :
            raise ValueError("Pinocchio model not initialized. Please call init_pin_model() first.")

        # Set the robot configuration to the neutral (home) configuration.
        # len = 9, 7 revolute + 2 prismatic
        self.model.q0 = pin.neutral(self.model)

        # Compute forward kinematics and update joint Jacobians.
        pin.forwardKinematics(self.model, self.data, self.model.q0)
        pin.computeJointJacobians(self.model, self.data, self.model.q0)
        pin.updateFramePlacements(self.model, self.data)

        # Retrieve the placement (as a pin.SE3 object) of the specified frame.
        M_pin = self.data.oMf[self.ee_frame_id]

        # the home homogeneous transformation matrix (4×4 numpy array).
        self.M = M_pin.homogeneous
        self.M_inv = pin.SE3.inverse(M_pin).homogeneous
        # self.M_inv = RobotModel.TransInv(self.M)   

    def calc_Slist_pin(self, joint_name=None):
        """
        Computes and returns the screw axis in the root (world) frame for a single joint.
        The joint is specified by its associated frame name. If no joint_name is provided,
        self.ee_frame_name is used.
        
        Parameters:
            joint_name (str): The name of the frame associated with the joint.
                              Defaults to self.ee_frame_name.
                              
        Returns:
            np.ndarray: A 6-element numpy array representing the screw axis of the specified joint.
        """
        self.calc_M_pin()
        # In Pinocchio, each frame is attached to a joint.
        joint_id = self.model.frames[self.ee_frame_id].parentJoint

        # Retrieve the joint Jacobian in the WORLD (space) frame.
        # For a 1-DoF joint, the Jacobian is a 6×1 matrix.
        J = pin.getJointJacobian(self.model, self.data, joint_id, pin.ReferenceFrame.WORLD)

        # self.print_formatted_nd_matrix(J)

        # exclude the finger
        # J = J[:, :len(self.rev_joint_list)]

        # [v,w]
        self.SScrewlist = J 
        # self.print_formatted_nd_matrix(self.SScrewlist)

    def calc_Blist_pin(self):

        self.calc_Slist_pin()
        # Compute the adjoint of the inverse transformation.
        Ad_M_inv = RobotModel.Adjoint(self.M_inv)

        # Transform the space-frame screw axes into the ee frame.
        self.Bscrewlist = Ad_M_inv @ self.SScrewlist
        # self.print_formatted_nd_matrix(self.Bscrewlist)

    class Controller:
        def __init__(self, modelpin: pin.Model,datapin: pin.Data):
            self.modelpin = modelpin
            self.datapin = datapin   
            self.ki = np.ones(modelpin.nv)*1       
            self.kp = np.ones(modelpin.nv)*600
            self.kd = np.ones(modelpin.nv)*200
            self.kdd = np.ones(modelpin.nv)

            self.q_error_inte = np.zeros(modelpin.nv)
            self.q_des = np.zeros(modelpin.nv)
            self.dq_des = np.zeros(modelpin.nv)
            self.ddq_des = np.zeros(modelpin.nv)
            self.tua_command = np.zeros(modelpin.nv)
            self.filtered_dq = np.zeros(modelpin.nv)
            print("Controller ee frame ID",self.modelpin.getFrameId("panda_hand"))
            self.dt = 0.3
            self.torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 200.0, 200.0])  # Example limits


            # Set up a logger for this class
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            # Create a file handler that logs even debug messages
            fh = logging.FileHandler(r'RobotPino\Denso_proj\controller.log', mode='w')
            fh.setLevel(logging.DEBUG)
            # Create a formatter and set it for the file handler
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            # Add the file handler to the logger
            self.logger.addHandler(fh)

        # inverse kinematics
        def IK_step(self, 
                    ee_frame_id: int, 
                    target_pose: pin.SE3, 
                    q_init: np.ndarray = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]),
                    tol: float = 1e-5,
                    max_iter: int = 300,
                    damping: float = 1e-10) -> np.ndarray:
            """
            Performs one IK routine to compute the desired joint configuration (q_des)
            that achieves the target end-effector pose.

            Args:
                ee_frame_id (int): The frame id of the end-effector.
                target_pose (pin.SE3): The desired end-effector pose.
                q_init (np.ndarray): The initial guess for the joint configuration.
                tol (float): Convergence tolerance for the error norm.
                max_iter (int): Maximum number of iterations.
                damping (float): Damping factor used in the pseudo-inverse computation.

            Returns:
                np.ndarray: The computed joint configuration (q_des) that (approximately) achieves the target pose.
            """
            q = q_init.copy()
            # Tuning parameters for error weighting:
            # You can increase w_rot if the rotational error is too large relative to the translation error.
            w_rot = 2.0      # Weight for the orientation error (first 3 components)
            w_trans = 1.0    # Weight for the translation error (last 3 components)
            # We use self.datapin and self.modelpin for the computations.

            for i in range(max_iter):
                # Update forward kinematics for current configuration
                pin.forwardKinematics(self.modelpin, self.datapin, q)
                pin.updateFramePlacements(self.modelpin, self.datapin)
                
                # Get current end-effector pose
                current_pose = self.datapin.oMf[ee_frame_id]
                # print("IK iteration of the pose homogeneous matrix:", current_pose.homogeneous)
                # 0.69888   0.03105   0.71456   0.38945
                # 0.04438  -0.99901  -0.00000  -0.00000
                # 0.71386   0.03171  -0.69957   0.45782
                # 0.00000   0.00000   0.00000   1.00000

                pose_des = current_pose.actInv(target_pose) # current_pose.inverse() * target_pose

                # # Compute error transformation: T_error = current_pose⁻¹ * target_pose
                # error_transform = current_pose.inverse() * target_pose
                # Get the 6D error (twist) using the logarithm map
                error_twist = pin.log6(pose_des).vector   # in joint frame
                # print("error_twist:", error_twist)

                # Apply separate weights to rotation and translation errors.
                error_twist[:3] *= w_rot
                error_twist[3:] *= w_trans

                # Format each element to six decimal places and join them into a single string
                logger_error_twist = ', '.join(f"{x:.6f}" for x in error_twist)
                self.logger.info(f"IK converged in {i} iterations with error twist : [{logger_error_twist}]")         
                
                error_norm = np.linalg.norm(error_twist)
                # Check convergence
                if error_norm < tol:
                    # print(f"IK converged in {i} iterations with error norm: {error_norm:.6f}")
                    self.logger.info(f"IK converged in {i} iterations with error norm: {error_norm:.6f}")
                    break
                
                # Compute the Jacobian of the end-effector frame.
                # Here we choose the LOCAL frame representation.
                # J = pin.computeFrameJacobian(self.modelpin, self.datapin, q , ee_frame_id, pin.ReferenceFrame.WORLD)
                J = pin.computeFrameJacobian(self.modelpin, self.datapin, q , ee_frame_id, pin.ReferenceFrame.LOCAL)
                # print("Local ee frame Jacobian:", J)
                
                # Compute the joint velocity update using a damped least-squares pseudo-inverse of the Jacobian.
                J_pinv = np.linalg.pinv(J, rcond=damping)
                dq = J_pinv.dot(error_twist)

                if (len(dq) > 6):
                    dq[0] = 0
            

                # Update configuration taking the manifold structure into account.
                q = pin.integrate(self.modelpin, q, dq * self.dt)

                if (len(q) > 6):
                    dq[0] = 0


                # if i % 10 == 0:
                #     print(f"Iteration {i}: error norm = {error_norm:.6f}")
            else:
                # print("IK did not converge within the maximum number of iterations.")
                self.logger.info("IK did not converge within the maximum number of iterations")
                
            return q
        


        
        # inverse dynamics
        def ID_step(self, des_q:np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray):# inverse dynamics

            """
            Computes the control torques for the Panda robot using a computed torque controller.

            Args:
                q (np.array): Current joint positions (shape: [9,])
                dq (np.array): Current joint velocities (shape: [9,])
                q_des (np.array): Desired joint positions (shape: [9,])
                dq_des (np.array): Desired joint velocities (shape: [9,])
                ddq_des (np.array): Desired joint accelerations (shape: [9,])
            
            Returns:
                np.array: Computed control torques (shape: [9,])
            """
            self.q_des = des_q
            
            # Compute the current mass matrix M(q) using the Composite Rigid Body Algorithm (CRBA)
            M = pin.crba(self.modelpin, self.datapin, q)

            # RNEA forward: position, velocity of each frame
            # RNEA backward: desired joint torque to generate desired ee wrench and acceleration

            # print("e: ", e)
            alpha = 0.9  # Smoothing factor (0 < alpha < 1)
            self.filtered_dq = alpha * self.filtered_dq + (1 - alpha) * dq

            # Compute the bias terms (gravity, Coriolis, centrifugal) using RNEA with zero acceleration
            # return is torque list: The computed torques needed at each joint to realize the motion
            tau_bias = pin.rnea(self.modelpin, self.datapin, q, self.filtered_dq, self.ddq_des)
            # tau_bias = pin.rnea(self.modelpin, self.datapin, q, self.dq_des, self.ddq_des)
            
            # Compute position and velocity errors
            e =  self.q_des - q
            if np.linalg.norm(e) < 0.1:
                self.q_error_inte += e


            ed = self.dq_des - self.filtered_dq

            # Compute the reference acceleration with PD feedback
            # ddq_ref = self.ddq_des + self.kd * ed + self.kp * e
            damping_factor = 0.1  # Adjust this value as needed
            ddq_ref = self.ddq_des + self.kd * ed + self.kp * e - damping_factor * dq + self.q_error_inte * self.ki

            # Inverse dynamics control law: tau = M(q) * ddq_ref + h(q, dq)
            self.tua_command = M.dot(ddq_ref) + tau_bias

            self.tua_command = np.clip(self.tua_command, -self.torque_limits, self.torque_limits)


            self.tau_error=M.dot(ddq_ref)
            self.tau_bias=tau_bias
            self.q_error=e

            # print("tau_error: ", M.dot(ddq_ref))
            # print("tau_bias: ", tau_bias)
            # print("tau_total: ", self.tua_command)
            # print("error: " ,e)
            
            formatted_tua = ', '.join(f"{x:.6f}" for x in self.tua_command)

            # self.logger.info(f"tua command : [{formatted_tua}]")    



        def reset(self): 
            self.q_des = np.zeros(self.modelpin.nv)
            self.dq_des = np.zeros(self.modelpin.nv)
            self.ddq_des = np.zeros(self.modelpin.nv)
            self.tua_command = np.zeros(self.modelpin.nv)
            self.q_error_inte = np.zeros(self.modelpin.nv)
            self.logger.info("Controller reset")



if __name__ == "__main__":
    
    panda_robot_urdf_file =r"C:\Users\13306\OneDrive\Coding_Proj\RobotPino\franka_description\robots\panda_arm_hand.urdf"
    package_dirs=r"C:\Users\13306\OneDrive\Coding_Proj\RobotPino"

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
                "ee_frame": "panda_hand",
                "root_frame": "panda_link0",
                "namespace": "panda",
            }
    
    panda_robot = RobotModel(config)
    panda_robot.load_desc_from_file(panda_robot_urdf_file, package_dirs)
    # notice the ee_frame is the end-effector joint frame,not link origin frame
    # panda_robot.cal_paras()
    # panda_robot.print_formatted_nd_matrix(panda_robot.Bscrewlist)


    # using Pinocchio
    panda_robot.init_pin_model()    

    panda_robot.calc_Slist_pin()
    panda_robot.calc_Blist_pin()

    panda_robot.print_formatted_nd_matrix(panda_robot.Bscrewlist)
    panda_robot.print_formatted_nd_matrix(panda_robot.M)

    # T = np.array([[1, 0,  0, 0],
    #                 [0, 0, -1, 0],
    #                 [0, 1,  0, 3],
    #                 [0, 0,  0, 1]])
    # se3 = panda_robot.MatrixLog6(T)
    # # print(se3)
    # # panda_robot.print_formatted_nd_matrix(se3)
    # M_SE3= panda_robot.MatrixExp6(se3)
    # panda_robot.print_formatted_nd_matrix(M_SE3)

    print("NOTE: In modern Robotics, there are Joint frame and link origin frame")
    print("NOTE: the joint {i} is the parent of the link {i} ")
    print("NOTE: if the last link prev joint frame is not the end-effector frame, define: last_link_to_ee_frame, may not be link origin frame")

    print("*****************************************************")

    print("Pinocchio: The link {i} mass property is attributed to the joint {i}")
    print("Pinocchio: the .model preserve all the frames, including the joint and link origin frames")