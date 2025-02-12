import scipy
import numpy as np
from urdf_parser_py.urdf import URDF
from packaging import version
from scipy.spatial.transform import Rotation as R
import modern_robotics as MR
import pinocchio as pin
import tempfile
"""
This module contains a class that can be used to calculate matrices required
for using the ModernRobotics Library. https://github.com/NxRLab/ModernRobotics
"""

class RobotModel(object):
    def __init__(self, configs):
        self.body_frame_name = configs["body_frame"]
        self.space_frame_name = configs["space_frame"]
        self.robot_namespace = configs.get("namespace")
        
        print("body frame link",self.body_frame_name)
        print("space frame link",self.space_frame_name)
        print("robot namespace",self.robot_namespace)

    def load_desc_from_param(self):
        """
        Gets the URDF from the robot_description parameter on the ROS parameter
        server
        """
        key = "%s/robot_description" % self.robot_namespace
        try:
            self.robot = URDF.from_parameter_server(key=key)
        except KeyError:
            print(
                (
                    "Error: `%s` not found on the ROS parameter server. "
                    "Check that it is loaded and under the right namespace."
                )
                % key
            )
            exit(1)

    def load_desc_from_file(self, file_path):
        """
        Gets the URDF from the robot's URDF file
        """
        self.urdf_file = file_path
        self.robot = URDF.from_xml_file(file_path)

    def load_desc_from_xml_string(self, xml_string):
        """
        Gets the URDF from an xml string containing the URDF
        """
        self.robot = URDF.from_xml_string(xml_string)

    def populate(self):
        """
        Populates relevant variables and lists
        """
        # Print all link names
        print("Links:")
        for link in self.robot.links:
            print(link.name)

        # Print all joint names
        print("\nJoints:")
        for joint in self.robot.joints:
            print(joint.name)
        print("\n")

        self.links = self.robot.links
        self.joints = self.robot.joints

        # init empty lists
        self.link_list = []
        self.joint_list = []
        self.rev_joint_list = []
        
        # joint frame
        self.M_joint_relative_trans_list = [] # home homogeneous matrix of each joint frame in the space frame SE(3), for kinematic chain only
        self.M_joint_space_trans_list = []
        # self.joint_pos_space_list = [] # world position of each joint frame
        # self.joint_rot_space_list = [] # world rotational matrix of each joint frame

        # link frame
        self.M_link_origin_space_frame_list = [] # home homogeneous matrix of each link origin frame (at CoM) in the space frame SE(3), for dynamics of the model only
        self.link_CoM_to_parent_joint_frame_list = [] # home homogeneous matrix of each link origin frame (at CoM) in the prev joint SE(3), for dynamics of the model only
        
        self.M_link_rot_body_axis_list = [] # the rotation axis list in the link frame, upper part of body screw in the link frame
        
        self.M_link_rot_space_axis_list = [] # the rotation axis list in the space frame, upper part of body screw in the link frame

        # Inertia properties
        self.link_com_list= [] # link centor of mass in prev joint frame (current link frame)      
        self.link_inertia_rpy_list= [] # frame of the inertia matrix in current link frame at center of mass
        self.link_mass_list= [] #mass
        self.link_inertia_matrix_list= [] # inertia matrix in current link frame at center of mass

        # find ee link in urdf tree
        self.body_frame = self._search(
            self.links, "%s" % (self.body_frame_name), "name"
        )

        self.space_frame = self._search(
            self.links, "%s" % (self.space_frame_name), "name"
        )

        if self.body_frame is None or self.space_frame is None:
            raise ValueError("Body Frame or Space Frame are empty")
        
        # for attr in dir(self.robot):
        #     if not attr.startswith("__"):  # Skip magic methods
        #         print(f"{attr}: {getattr(self.robot, attr)}")

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
        self.last_link_to_ee_frame=np.array(last_link_to_ee_frame)

        # Convert lists to NumPy arrays
        self.M_joint_relative_trans_list = np.array(self.M_joint_relative_trans_list) # shape: (n, 4, 4) if there are n transforms
        self.M_joint_space_trans_list = np.array(self.M_joint_space_trans_list) # shape: (n, 4, 4) if there are n transforms

        self.M_link_origin_space_frame_list = np.array(self.M_link_origin_space_frame_list) # shape: (n, 4, 4) if there are n transforms
        self.link_CoM_to_parent_joint_frame_list = np.array(self.link_CoM_to_parent_joint_frame_list) # shape: (n, 4, 4) if there are n transforms

        self.M_link_rot_body_axis_list = np.array(self.M_link_rot_body_axis_list) # shape: (n, 3) in body frame
        self.M_link_rot_space_axis_list = np.array(self.M_link_rot_space_axis_list) # shape: (n, 3) in space frame

        self.link_com_list = np.array(self.link_com_list) # shape depends on collected COMs
        self.link_inertia_rpy_list = np.array(self.link_inertia_rpy_list)
        self.link_mass_list = np.array(self.link_mass_list)
        self.link_inertia_matrix_list = np.array(self.link_inertia_matrix_list)


        self.calc_M() # home-config homogeneous matrix of end-effector SE(3), space frame
        self.calc_Alist() # home-config screw axis of the end-effector in current link (prev joint) frame
        self.calc_Slist() # home-config screw axis of the end-effector in space frame
        self.calc_Blist() # home-config screw axis of the end-effector

        self._print_robot_info()

    def calc_M(self):
        """
        Calculates the homogeneous matrix describing the pose of the end
            effector in SE(3)
        """
        M = np.eye(4, dtype="float64")
        for T in self.M_joint_relative_trans_list:
            M = np.dot(M, T)
        self.M =np.dot(M, self.last_link_to_ee_frame)

    def calc_Alist(self):
        """
        Calculates the list of screw axes
        """
        num_axes = len(self.rev_joint_list)  # number of revolute axes defines Slist shape
        Ascrew = np.zeros((6, num_axes))
        for i in range(num_axes):
            if self.joint_list[i] in self.rev_joint_list:
                w = np.array(self.M_link_rot_body_axis_list[i], dtype="float64")
                # print("the parent joint in the link origin frame: ", -self.link_CoM_to_parent_joint_frame_list[i, :3, 3])
                q = np.array(-self.link_CoM_to_parent_joint_frame_list[i, :3, 3], dtype="float64")
                v = np.cross(-w, q)
                Ascrew[:, i] = np.hstack((w, v))
        self.Ascrewlist = Ascrew # the body screw of prev joint with respect to each link origin frame

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
                Sscrew[:, i] = np.hstack((w, v))
        self.Sscrewlist = Sscrew

    def calc_Blist(self):
        """
        Calculates the list of screw axes in the end-effector frame from the space screw
        """
        num_axes = len(self.rev_joint_list) # number of revolute axes defines Slist shape
        M_inv = MR.TransInv(self.M)
        ee_AdT_inv = MR.Adjoint(M_inv)
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
        current_link = self.body_frame
        parent_joint = self._get_parent_joint(self.body_frame)


        # iterate backwards through link/joint tree up to base
        while current_link is not None and (current_link.name != self.space_frame.name):

            print("Joint name: ",parent_joint.name)

            self.link_list.insert(0, current_link)
            self.joint_list.insert(0, parent_joint)

            # keep track of revolute joints and their axes
            if parent_joint.type == "revolute":
                self.rev_joint_list.insert(0, parent_joint)

            self.M_joint_relative_trans_list.insert(0, self._get_parent_transform(current_link))

            if self._get_parent_axis(current_link): # if not none
                self.M_link_rot_body_axis_list.insert(0, self._get_parent_axis(current_link))

            else: # fixed joint default axis
                self.M_link_rot_body_axis_list.insert(0, [0.0, 0.0, 1.0])

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
        # print("The M_axis_list is",self.M_link_rot_body_axis_list)
        # print("The link_com_list is",self.link_com_list)

        M_joint_space_trans = np.eye(4, dtype="float64")
        for i in range(len(self.joint_list)):
            # joint frame 
            M_joint_space_trans = np.dot(M_joint_space_trans, self.M_joint_relative_trans_list[i])
            self.M_joint_space_trans_list.append(M_joint_space_trans)
            
            # link frame
            self.M_link_rot_space_axis_list.append(np.dot(M_joint_space_trans[:3, :3], self.M_link_rot_body_axis_list[i]))

            link_CoM_to_parent_joint_frame = self._get_link_CoM_transform(self.link_com_list[i], self.link_inertia_rpy_list[i])
            
            self.link_CoM_to_parent_joint_frame_list.append(link_CoM_to_parent_joint_frame)
            self.M_link_origin_space_frame_list.append(np.dot(M_joint_space_trans,link_CoM_to_parent_joint_frame))


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

        print("self.M_joint_space_trans_list (joint transformation matrix of SE(3) expressed in the space frame): \n")
        self.print_formatted_nd_matrix(self.M_joint_space_trans_list)
              
        print("self.M_link_origin_space_frame_list (each link origin frame (at CoM) transformation matrix of SE(3) expressed in the space frame): \n")
        self.print_formatted_nd_matrix(self.M_link_origin_space_frame_list)

        print("self.link_CoM_to_parent_joint_frame_list (each link origin frame (at CoM) transformation matrix of SE(3) expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_CoM_to_parent_joint_frame_list)

        print("self.M_link_rot_body_axis_list (axis of each revolute joint expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.M_link_rot_body_axis_list)
        
        print("self.M_link_rot_space_axis_list (axis of each revolute joint expressed in the space frame): \n")
        self.print_formatted_nd_matrix(self.M_link_rot_space_axis_list)

        print("self.link_com_list (CoM of each link expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_com_list)

        print("self.link_inertia_rpy_list (Roll-Pitch-Yaw of each link expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_inertia_rpy_list)

        print("self.link_mass_list (mass of each link expressed in the prev joint frame): \n")
        self.print_formatted_nd_matrix(self.link_mass_list)

        print("self.link_inertia_matrix_list (Inertia matrix of each link expressed in the current link frame, i.e., current joint frame): \n")
        self.print_formatted_nd_matrix(self.link_inertia_matrix_list)

        print("self.Ascrewlist (space frame screw): \n")
        self.print_formatted_nd_matrix(self.Ascrewlist)

        print("self.Sscrewlist (link frame screw): \n")
        self.print_formatted_nd_matrix(self.Sscrewlist)

        print("self.Bscrewlist (space frame screw): \n")
        self.print_formatted_nd_matrix(self.Bscrewlist)


        print("self.M (home config homogeneous matrix of SE(3)): \n")
        self.print_formatted_nd_matrix(self.M)

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
    def VecToso3(omg):
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
    def so3ToVec(so3mat):
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
    def RpToTrans(R, p):
        """Converts a rotation matrix and a position vector into homogeneous
        transformation matrix

        :param R: A 3x3 rotation matrix
        :param p: A 3-vector
        :return: A homogeneous transformation matrix corresponding to the inputs

        Example Input:
            R = np.array([[1, 0,  0],
                          [0, 0, -1],
                          [0, 1,  0]])
            p = np.array([1, 2, 5])
        Output:
            np.array([[1, 0,  0, 1],
                      [0, 0, -1, 2],
                      [0, 1,  0, 5],
                      [0, 0,  0, 1]])
        """
        return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

    @staticmethod
    def TransToRp(T):
        """Converts a homogeneous transformation matrix into a rotation matrix
        and position vector

        :param T: A homogeneous transformation matrix
        :return: (R, p) where R is a 3x3 rotation matrix and p is a 3-vector

        Example Input:
            T = np.array([[1, 0,  0, 0],
                          [0, 0, -1, 0],
                          [0, 1,  0, 3],
                          [0, 0,  0, 1]])
        Output:
            (np.array([[1, 0,  0],
                       [0, 0, -1],
                       [0, 1,  0]]),
             np.array([0, 0, 3]))
        """
        T = np.array(T)
        return T[0:3, 0:3], T[0:3, 3]

    @staticmethod
    def TransInv(T):
        """Inverts a homogeneous transformation matrix

        :param T: A homogeneous transformation matrix
        :return: The inverse of T

        Example Input:
            T = np.array([[1, 0,  0, 0],
                          [0, 0, -1, 0],
                          [0, 1,  0, 3],
                          [0, 0,  0, 1]])
        Output:
            np.array([[1,  0, 0,  0],
                      [0,  0, 1, -3],
                      [0, -1, 0,  0],
                      [0,  0, 0,  1]])
        """
        R, p = RobotModel.TransToRp(T)
        Rt = np.array(R).T
        return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

    @staticmethod
    def VecTose3(w,v):
        """Converts a spatial velocity vector into a 4x4 matrix in se3

        :param V: A 6-vector representing a spatial velocity
        :return: The 4x4 se3 representation of V

        Example Input:
            V = np.array([1, 2, 3, 4, 5, 6])
        Output:
            np.array([[ 0, -3,  2, 4],
                      [ 3,  0, -1, 5],
                      [-2,  1,  0, 6],
                      [ 0,  0,  0, 0]])
        """
        return np.r_[np.c_[RobotModel.VecToso3([w[0], w[1], w[2]]), [v[0], v[1], v[2]]],
                     np.zeros((1, 4))]

    @staticmethod
    def se3ToVec(se3mat):
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
        return np.r_[ [se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                      [se3mat[0][3], se3mat[1][3], se3mat[2][3]] ]

    @staticmethod
    def ScrewToAxis(q, s, h):
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
        return np.r_[s, np.cross(q, s) + h * s]

    @staticmethod
    def AxisAng6(expc6):
        """Converts a 6-vector of exponential coordinates into screw axis-angle
        form

        :param expc6: A 6-vector of exponential coordinates for rigid-body motion S*theta
        :return: (normalized screw axis, theta)

        Example Input:
            expc6 = np.array([1, 0, 0, 1, 2, 3])
        Output:
            (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
        """
        theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
        if RobotModel.NearZero(theta):
            theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
        return (np.array(expc6) / theta, theta)

    @staticmethod
    def MatrixExp3(so3mat):
        """Computes the matrix exponential of a matrix in so(3)

        :param so3mat: A 3x3 skew-symmetric matrix
        :return: The matrix exponential of so3mat

        Example Input:
            so3mat = np.array([[ 0, -3,  2],
                               [ 3,  0, -1],
                               [-2,  1,  0]])
        Output:
            np.array([[-0.69492056,  0.71352099,  0.08929286],
                      [-0.19200697, -0.30378504,  0.93319235],
                      [ 0.69297817,  0.63134970,  0.34810748]])
        """
        w = RobotModel.so3ToVec(so3mat)
        return pin.exp3(w)

    @staticmethod
    def MatrixLog3(R):
        """Computes the matrix logarithm of a rotation matrix

        :param R: A 3x3 rotation matrix
        :return: The matrix logarithm of R

        Example Input:
            R = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
        Output:
            np.array([[          0, -1.20919958,  1.20919958],
                      [ 1.20919958,           0, -1.20919958],
                      [-1.20919958,  1.20919958,           0]])
        """
        w = pin.log3(R)

        return RobotModel.VecToso3(w)
    
    @staticmethod
    def Adjoint(T):
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
        R, p = RobotModel.TransToRp(T)
        return np.r_[np.c_[R, np.zeros((3, 3))],
                      np.c_[np.dot(RobotModel.VecToso3(p), R), R]]

    @staticmethod
    def MatrixExp6(se3mat):
        """
        Computes the matrix exponential of a 4x4 se(3) matrix to yield an SE(3)
        homogeneous transformation matrix.

        Parameters:
            se3mat (np.ndarray): A 4x4 matrix representing an element of se(3),
                                i.e. a twist in matrix form:
                                    [ hat(ω)   v ]
                                    [   0      0 ]
                                (typically ω is scaled by an angle).

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix in SE(3).
        """
        # Convert the 4x4 se(3) matrix into a 6D twist vector.
        xi = RobotModel.se3ToVec(se3mat) # [w,v] 6d
        xi[:3], xi[3:] = xi[3:].copy(), xi[:3].copy()

        # Return the 4x4 homogeneous matrix.
        return pin.exp6(xi).homogeneous

    
    @staticmethod
    def MatrixLog6(T):
        """
        Computes the logarithm of a 4x4 se(3) matrix to yield a 6D twist vector.
        
        Parameters:
            T (np.ndarray): A 4x4 matrix representing an element of SE(3),
                                i.e. a homogeneous transformation matrix.
        
        Returns:
            np.ndarray: A 6D twist vector representing the logarithm of T.
        """
        T_pin = pin.SE3(T)
        Screw = pin.log6(T_pin)

        return RobotModel.VecTose3(Screw.angular,Screw.linear)

    # def MatrixExp6(se3mat):
    # def MatrixLog6(T):
    # def ProjectToSO3(mat):
    # def ProjectToSE3(mat):
    # def DistanceToSO3(mat):
    # def DistanceToSE3(mat):
    # def TestIfSO3(mat):
    # def TestIfSE3(mat):

    # Pinocchio lib based fonctions

    def init_pin_model(self):
        """
        Initializes the Pinocchio model and data using the current URDF robot description.
        The Pinocchio model is built from the URDF XML string.
        """
        # Ensure the URDF has been loaded
        if not hasattr(self, "robot"):
            raise ValueError("Robot URDF not loaded. Please load a URDF description first using load_desc_from_file, load_desc_from_param, or load_desc_from_xml_string.")
        
        # Convert the URDF object to an urdf file
        try:
            urdf_xml = self.robot.to_xml_string()
        except AttributeError:
            raise AttributeError("The URDF object does not have a 'to_xml_string' method. Ensure your URDF is in the correct format.")

        # Write the URDF string to a temporary file.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.urdf') as tmp_file:
            tmp_file.write(urdf_xml)
            tmp_filename = tmp_file.name

        # Build the Pinocchio model and create its data object
        self.pin_model = pin.buildModelFromUrdf(tmp_filename)
        self.pin_data = self.pin_model.createData()

        print("Pinocchio model and data have been successfully initialized.")

    def get_joint_path_pin(self):
        """
        Computes the joint path (chain) from the space frame to the body (end-effector) frame.
        Returns:
            list[int]: A list of joint IDs (excluding the base fixed joint) along the kinematic chain.
        """
        # Get the frame IDs for the space and body frames.
        base_frame_id = self.pin_model.getFrameId(self.space_frame_name)
        ee_frame_id   = self.pin_model.getFrameId(self.body_frame_name)

        # Each frame in Pinocchio is attached to a joint. Retrieve the parent joint IDs.
        base_joint_id = self.pin_model.frames[base_frame_id].parent
        ee_joint_id   = self.pin_model.frames[ee_frame_id].parent

        # Build the chain by traversing from the end-effector joint to the base joint.
        chain = []
        j = ee_joint_id
        # Traverse until we reach the base joint.
        while j != base_joint_id:
            chain.append(j)
            j = self.pin_model.parents[j]
        # Reverse the chain so that it goes from base to end-effector.
        chain.reverse()
        return chain

    def calc_M_pin(self, joint_name=None):
        """
        Computes and returns home config SE(3) in the space (world) frame for a single joint.
        The joint is specified by its associated frame name. If no joint_name is provided,
        self.body_frame_name is used.
        """
        if joint_name is None:
            joint_name = self.body_frame_name

        # Ensure the Pinocchio model and data have been initialized.
        if not hasattr(self, 'pin_model') or not hasattr(self, 'pin_data'):
            raise ValueError("Pinocchio model not initialized. Please call init_pinocchio_model() first.")

        # Set the robot configuration to the neutral (home) configuration.
        q0 = pin.neutral(self.pin_model)

        # Compute forward kinematics and update joint Jacobians.
        pin.forwardKinematics(self.pin_model, self.pin_data, q0)
        pin.computeJointJacobians(self.pin_model, self.pin_data, q0)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # Retrieve the frame ID for the given joint name.
        try:
            self.ee_frame_id = self.pin_model.getFrameId(joint_name)
        except Exception as e:
            raise ValueError("Frame with name '{}' not found in the Pinocchio model.".format(self.ee_frame_id))

        # Retrieve the placement (as a pin.SE3 object) of the specified frame.
        M_pin = self.pin_data.oMf[self.ee_frame_id]
        #the homogeneous transformation matrix (4×4 numpy array).
        self.M = M_pin.homogeneous
        self.M_inv = pin.SE3.inverse(M_pin).homogeneous
        # self.M_inv = RobotModel.TransInv(self.M)   

    def calc_Slist_pin(self, joint_name=None):
        """
        Computes and returns the screw axis in the space (world) frame for a single joint.
        The joint is specified by its associated frame name. If no joint_name is provided,
        self.body_frame_name is used.
        
        Parameters:
            joint_name (str): The name of the frame associated with the joint.
                              Defaults to self.body_frame_name.
                              
        Returns:
            np.ndarray: A 6-element numpy array representing the screw axis of the specified joint.
        """

        self.calc_M_pin()
        # In Pinocchio, each frame is attached to a joint.
        joint_id = self.pin_model.frames[self.ee_frame_id].parentJoint

        # Retrieve the joint Jacobian in the WORLD (space) frame.
        # For a 1-DoF joint, the Jacobian is a 6×1 matrix.
        J = pin.getJointJacobian(self.pin_model, self.pin_data, joint_id, pin.ReferenceFrame.WORLD)

        # self.print_formatted_nd_matrix(J)

        # exclude the finger
        J = J[:, :len(self.rev_joint_list)]

        # [w,v]
        self.SScrewlist = np.vstack([J[3:6], J[0:3]])

        # [v,w]
        # self.SScrewlist = J 
        # self.print_formatted_nd_matrix(self.SScrewlist)

    def calc_Blist_pin(self):

        self.calc_Slist_pin()
        # Compute the adjoint of the inverse transformation.
        Ad_M_inv = RobotModel.Adjoint(self.M_inv)

        # Transform the space-frame screw axes into the body frame.
        self.Bscrewlist = Ad_M_inv @ self.SScrewlist
        # self.print_formatted_nd_matrix(self.Bscrewlist)


