"""
@file sim_robot_interface.py
@package mjcsim
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

import numpy as np
import mujoco as mjc
from pathlib import Path
import xml.etree.ElementTree as ET

from mjcsim.joint_controller import JointController

class SimRobotInterface:
    def __init__(self, rob_setting):
        """Initializes the simulation interface of a robot.

        Args:
            rob_setting: a container for configuration parameters.
        """
        self.xml_filename = rob_setting.xml_filename
        self.use_fixed_base = rob_setting.use_fixed_base
        self.limb_names = rob_setting.limb_names
        self.limb_endeff_names = rob_setting.limb_endeff_names
        self.joint_names = rob_setting.joint_names
        self.base_init_pos = rob_setting.base_init_pos
        self.base_init_vel = rob_setting.base_init_vel
        self.joint_init_pos = rob_setting.joint_init_pos
        self.joint_init_vel = rob_setting.joint_init_vel

        assert len(self.limb_names) == len(self.limb_endeff_names)
        self.limb_name_endeff_name_map = {}
        self.endeff_name_limb_name_map = {}
    
        for idx in range(len(self.limb_names)):
            self.limb_name_endeff_name_map[
                self.limb_names[idx]] = self.limb_endeff_names[idx]
            self.endeff_name_limb_name_map[
                self.limb_endeff_names[idx]] = self.limb_names[idx]

        if self.use_fixed_base:
            self.nq = len(self.joint_names)
            self.nv = len(self.joint_names)
            self.nj = len(self.joint_names)
            self.ne = len(self.limb_endeff_names)
        else:
            self.nq = len(self.base_init_pos) + len(self.joint_init_pos)
            self.nv = len(self.base_init_vel) + len(self.joint_init_vel)
            self.nj = len(self.joint_names)
            self.ne = len(self.limb_endeff_names)

        self.base_lin_vel_prev = None
        self.base_ang_vel_prev = None
        self.base_lin_acc = np.zeros(3)
        self.base_ang_acc = np.zeros(3)

        self.g = np.array([0., 0., -9.81])

        self.rng = np.random.default_rng()

        self.init_orn_inv = np.array([1.0, 0.0, 0.0, 0.0])
        self.joint_qpos_indices = []
        self.joint_dof_indices = []
        self.joint_states = []

        # IMU
        self.imu_lin_acc = np.zeros(3)
        self.imu_ang_vel = np.zeros(3)
        self.base_orn = np.array([1.0, 0.0, 0.0, 0.0])
        self.base_pos = np.array(self.base_init_pos[:3]) if len(self.base_init_pos) >= 3 else np.zeros(3)
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)

        self.base_imu_acc_bias = np.zeros(3)
        self.base_imu_gyro_bias = np.zeros(3)
        self.base_imu_acc_thermal = np.zeros(3)
        self.base_imu_gyro_thermal = np.zeros(3)
        self.base_imu_acc_bias_noise = 0.0001          # m/(sec^3*sqrt(Hz))
        self.base_imu_gyro_bias_noise = 0.0000309      # rad/(sec^2*sqrt(Hz))
        self.base_imu_acc_thermal_noise = 0.00001962   # m/(sec^2*sqrt(Hz))
        self.base_imu_gyro_thermal_noise = 0.000000873 # rad/(sec*sqrt(Hz))

        self.joint_torque_limits = []
        self.joint_torque_limits = self._get_torque_limits_from_xml()
        self.joint_controller = JointController(self.joint_names, self.joint_torque_limits)
    

    def compute_numerical_quantities(self, dt=0.001):
        """Compute numerical robot quantities from simulation results."""
        
        # Dynamically obtain the root joint, rather than hard-coding it
        root_jnt_id = self._get_root_joint_id()
        
        if root_jnt_id is not None:
            root_dof_start = self.model.jnt_dofadr[root_jnt_id]
            # Obtain basic position and direction
            self.base_pos = self.data.qpos[root_dof_start:root_dof_start+3].copy()
            base_orn_w = self.data.qpos[root_dof_start+3:root_dof_start+7].copy()  
            # Obtain basic linear velocity and angular velocity
            root_vel_start = self.model.jnt_dofadr[root_dof_start]
            base_lin_vel_w = self.data.qvel[root_vel_start:root_vel_start+3].copy()
            base_ang_vel_w = self.data.qvel[root_vel_start+3:root_vel_start+6].copy()
        else:
            # If there is no root joint (fixed base), use the default value
            self.base_pos = np.array(self.base_init_pos)
            base_orn_w = np.array([1.0, 0.0, 0.0, 0.0])  # unit quaternion
            base_lin_vel_w = np.zeros(3)
            base_ang_vel_w = np.zeros(3)

        # Obtain joint status
        joint_pos = self.data.qpos[self.joint_qpos_indices].copy()
        joint_vel = self.data.qvel[self.joint_dof_indices].copy()
        # Create a compatible joint_states structure
        self.joint_states = [(pos, vel, np.zeros(6), 0.0) for pos, vel in zip(joint_pos, joint_vel)]
        # Calculate the pose relative to the initial direction
        init_orn_inv = np.array([self.init_orn_inv[0], self.init_orn_inv[1], 
                                self.init_orn_inv[2], -self.init_orn_inv[3]])  # quaternion conjugate
        self.base_orn = self._multiply_quaternions(base_orn_w, init_orn_inv)

        # Transform velocity to base frame
        self.base_lin_vel = self._transform_velocity_to_base_frame(
            base_lin_vel_w, self.base_orn)
        self.base_ang_vel = self._transform_velocity_to_base_frame(
            base_ang_vel_w, self.base_orn)

        # Calculate the basic acceleration
        if self.base_lin_vel_prev is not None and self.base_ang_vel_prev is not None:
            self.base_lin_acc = (1.0/dt) * (self.base_lin_vel - self.base_lin_vel_prev)
            self.base_ang_acc = (1.0/dt) * (self.base_ang_vel - self.base_ang_vel_prev)
        
        self.base_lin_vel_prev = self.base_lin_vel
        self.base_ang_vel_prev = self.base_ang_vel

        # Calculate IMU values
        self.imu_lin_acc = (self.base_lin_acc - 
            self._transform_velocity_to_base_frame(self.g, self.base_orn))
        self.imu_ang_vel = self.base_ang_vel

        # Update IMU bias and noise
        self.base_imu_acc_bias += dt * (
            self.base_imu_acc_bias_noise/np.sqrt(dt)) * self.rng.standard_normal(3)
        self.base_imu_gyro_bias += dt * (
            self.base_imu_gyro_bias_noise/np.sqrt(dt)) * self.rng.standard_normal(3)

        self.base_imu_acc_thermal += (
            self.base_imu_acc_thermal_noise/np.sqrt(dt)) * self.rng.standard_normal(3)
        self.base_imu_gyro_thermal += (
            self.base_imu_gyro_thermal_noise/np.sqrt(dt)) * self.rng.standard_normal(3)

    def _get_root_joint_id(self):
        """Obtain the root joint ID and handle the fixed base condition"""
        if self.use_fixed_base:
            return None  # The fixed base has no root joint
        
        # Try different root joint names
        root_joint_names = ['base_link', 'root', 'world', 'floating_base', 'base_joint', 'robot_base']
        for name in root_joint_names:
            try:
                # First, try to find the body with this name
                body_id = mjc.mj_name2id(self.model, mjc.mjtObj.mjOBJ_BODY, name)
                # Then find the freejoint in this body
                # Look for joints attached to this body
                for joint_id in range(self.model.njnt):
                    if self.model.jnt_bodyid[joint_id] == body_id:
                        if self.model.jnt_type[joint_id] == mjc.mjtJoint.mjJNT_FREE:
                            return joint_id
            except:
                continue
        
        # If the named root joint is not found, search for the first joint of degree of freedom type
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mjc.mjtJoint.mjJNT_FREE:
                return joint_id
        
        return None

    def _multiply_quaternions(self, q1, q2):
        """quaternion multiplication"""
        result = np.zeros(4)
        mjc.mju_mulQuat(result, q1, q2)
        return result
    
    def _transform_velocity_to_base_frame(self, velocity_world, orientation):
        """Transform velocity from world frame to base frame
        Args:
            velocity_world: velocity vector in world frame
            orientation: current orientation quaternion [w, x, y, z]
        Returns:
            velocity vector in base frame
        """
        # Construct a rotation matrix from quaternion
        rotation_matrix = self._quat_to_rot(orientation)
        
        # Velocity conversion: R^T * v_world (or R^-1 * v_world, for the rotation matrix R^T = R^-1)
        velocity_base = rotation_matrix.T @ velocity_world
        
        return velocity_base

    def _quat_to_rot(self, quat):
        """Convert quaternion to rotation matrix"""
        rotation_matrix = np.zeros(9)
        # Use the function provided by MuJoCo to convert quaternion into rotation matrix
        mjc.mju_quat2Mat(rotation_matrix, quat)
        return rotation_matrix.reshape(3, 3)

    def _rot_to_quat(self, rot_matrix):
        """Convert rotation matrix to quaternion"""
        quat = np.zeros(4)
        mjc.mju_mat2Quat(quat, rot_matrix.flatten())
        return quat
            
    def _get_torque_limits_from_xml(self):
        """Read joint torque limits from the XML file
        
        Returns:
            list: Joint torque limits list
        """
        # Parse XML file
        tree = ET.parse(self.xml_filename)
        root = tree.getroot()
        torque_limits = []
        
        for joint_name in self.joint_names:
            torque_limit = 100.0  # Default torque limit
            # Find the force limit corresponding to the joint in XML
            for joint in root.findall('.//joint[@name]'):
                    if joint.get('name') == joint_name:
                        actuatorfrcrange_str = joint.get('actuatorfrcrange')
                        if actuatorfrcrange_str:
                            actuatorfrcrange = list(map(float, actuatorfrcrange_str.split()))
                            # Use the maximum absolute value of actuatorfrcrange as the torque limit
                            torque_limit = max(abs(val) for val in actuatorfrcrange)
                        break
            torque_limits.append(torque_limit)
        
        return torque_limits
    
    def attach_to_sim(self, model, data, robot_name=None):
        """Attach the robot interface to a MuJoCo simulation."""
        self.model = model
        self.data = data
        
        # Get Joint ID
        self.joint_ids = []
        for joint_name in self.joint_names:
            try:
                joint_id = mjc.mj_name2id(self.model, mjc.mjtObj.mjOBJ_JOINT, joint_name)
                self.joint_ids.append(joint_id)
            except:
                print(f"Warning: Joint {joint_name} not found in model")
        
        self.joint_ids = np.array(self.joint_ids)

        # Initialize the applied joint torques
        self.applied_joint_torques = np.zeros(len(self.joint_names))
        
        # Set joint position and degree of freedom index
        self.joint_qpos_indices = []
        self.joint_dof_indices = []
        for joint_id in self.joint_ids:
            qpos_start = self.model.jnt_qposadr[joint_id]
            dof_start = self.model.jnt_dofadr[joint_id]
            
            # Determine the number of degrees of freedom based on the joint type
            joint_type = self.model.jnt_type[joint_id]
            if joint_type == mjc.mjtJoint.mjJNT_HINGE or joint_type == mjc.mjtJoint.mjJNT_SLIDE:
                # Single degree of freedom joint
                self.joint_qpos_indices.append(qpos_start)
                self.joint_dof_indices.append(dof_start)
            elif joint_type == mjc.mjtJoint.mjJNT_FREE:
                # 6-degree-of-freedom free joint
                self.joint_qpos_indices.extend([qpos_start + i for i in range(7)])  # 3 positions + 4 quaternions
                self.joint_dof_indices.extend([dof_start + i for i in range(6)])    # 3 linear velocities + 3 angular velocities

        self.joint_qpos_indices = np.array(self.joint_qpos_indices)
        self.joint_dof_indices = np.array(self.joint_dof_indices)

        self.joint_states = [(0.0, 0.0, np.zeros(6), 0.0) for _ in range(len(self.joint_names))]

    def get_joint_positions(self):
        """Get the robot's joint positions.

        Returns:
            joint_positions (ndarray): Joint positions.
        """
        joint_pos = []
        for idx, jn in enumerate(self.joint_names):
            joint_pos.append(self.joint_states[idx][0])
        return np.array(joint_pos)
    
    def get_joint_velocities(self):
        """Get the robot's joint velocities.

        Returns:
            joint_velocities (ndarray): Joint velocities.
        """
        joint_vel = []
        for idx, jn in enumerate(self.joint_names):
            joint_vel.append(self.joint_states[idx][1])
        return (np.array(joint_vel))

    
    def apply_joint_actions(self, actions):
        """Apply the desired action to the joints.

        Args:
            actions (ndarray): Joint action to be applied.
        """
        # Calculate joint torque
        joint_torque = self.joint_controller.convert_to_torque(
            self.get_joint_positions(), 
            self.get_joint_velocities(), 
            actions
        )
        
        # Update the applied joint torque
        self.applied_joint_torques = joint_torque
        
        for i, joint_name in enumerate(self.joint_names):
            if i < len(self.data.ctrl):
                self.data.ctrl[i] = joint_torque[i]

    def get_joint_efforts(self):
        """Get the robot's joint forces (prismatic) and torques (revolute).

        Returns: 
            joint_efforts (ndarray): Joint forces and torques.
        """
        return self.applied_joint_torques
       
    def get_limb_contact_forces(self):
        """Get robot's limb contact forces with the environment in MuJoCo.

        Returns:
            contact_forces (ndarray): limb contact forces in normal direction.
        """
        contact_forces = []
        for endeff_name in self.limb_endeff_names:
            endeff_id = self.model.geom(endeff_name).id
            normal_force = 0.
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == endeff_id or contact.geom2 == endeff_id:
                    force = np.zeros(6)  # [3 for force, 3 for torque]
                    mjc.mj_contactForce(self.model, self.data, i, force)
                    normal_force += abs(force[0])  
            contact_forces.append(normal_force)
        
        return np.array(contact_forces)
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_raw_base_imu_angular_velocity(self):
        """Get polluted base IMU gyroscope angular velocity. Assume that 
        the IMU coordinate coincides with the robot's body coordinate.

        Returns:
            np.array((3,1)) IMU gyroscope angular velocity (base frame).
        """
        return (self.imu_ang_vel + self.base_imu_gyro_bias + 
            self.base_imu_gyro_thermal + 0.015*self.rng.standard_normal(3))
    
    def get_base_imu_angular_velocity(self):
        """Get simulated base IMU gyroscope angular velocity. Assume that 
        the IMU coordinate coincides with the robot's body coordinate.

        Returns:
            np.array((3,1)) IMU gyroscope angular velocity (base frame).
        """
        return (self.imu_ang_vel + self.base_imu_gyro_bias + 
            self.base_imu_gyro_thermal)
    
    def get_raw_base_imu_linear_acceleration(self):
        """Get polluted base IMU accelerometer acceleration. Assume that 
        the IMU coordinate coincides with the robot's body coordinate.

        Returns:
            np.array((3,1)) IMU accelerometer acceleration (base frame, gravity offset).
        """
        return (self.imu_lin_acc + self.base_imu_acc_bias + 
            self.base_imu_acc_thermal + 0.015*self.rng.standard_normal(3))

    def get_base_imu_linear_acceleration(self):
        """Get simulated base IMU accelerometer acceleration. Assume that 
        the IMU coordinate coincides with the robot's body coordinate.

        Returns:
            np.array((3,1)) IMU accelerometer acceleration (base frame, gravity offset).
        """
        return (self.imu_lin_acc + self.base_imu_acc_bias + 
            self.base_imu_acc_thermal)

    def get_raw_joint_positions(self):
        """Get the robot's joint positions polluted by noise.

        Returns:
            joint_positions (ndarray): Joint positions.
        """
        joint_pos = []
        for idx, jn in enumerate(self.joint_names):
            joint_pos.append(self.joint_states[idx][0])
        return (np.array(joint_pos) + 0.005*self.rng.standard_normal(self.nj))
    
    def get_raw_joint_velocities(self):
        """Get the robot's joint velocities polluted by noise.

        Returns:
            joint_velocities (ndarray): Joint velocities.
        """
        joint_vel = []
        for idx, jn in enumerate(self.joint_names):
            joint_vel.append(self.joint_states[idx][1])
        return (np.array(joint_vel) + 0.15*self.rng.standard_normal(self.nj))
    
    def get_raw_joint_efforts(self):
        """Get the robot's joint forces (prismatic) and torques (revolute) polluted 
        by noise.

        Returns: 
            joint_efforts (ndarray): Joint forces and torques.
        """
        joint_eft = self.applied_joint_torques + self.rng.standard_normal(self.nj)
        return joint_eft
    
    def get_raw_limb_contact_forces(self):
        """Get robot's limb contact forces with the environment in MuJoCo.

        Returns:
            contact_forces (ndarray): limb contact forces in normal direction.
        """
        contact_forces = []
        for endeff_name in self.limb_endeff_names:
            try:
                endeff_id = mjc.mj_name2id(self.model, mjc.mjtObj.mjOBJ_GEOM, endeff_name)
            except:
                print(f"Warning: Geometry {endeff_name} not found in model")
                contact_forces.append(0.)
                continue
            
            normal_force = 0.
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == endeff_id or contact.geom2 == endeff_id:
                    force = np.zeros(6)
                    mjc.mj_contactForce(self.model, self.data, i, force)
                    normal_force += abs(force[0])
            contact_forces.append(normal_force)
        
        return np.array(contact_forces) + self.rng.standard_normal(self.ne)
    
    def get_base_position(self):
        """Get the robot's base position.

        Returns:
            base_pos (ndarray): base position.
        """
        return np.array(self.base_pos)

    def get_base_rotation(self):
        """Get the robot's base orientation in rotation matrix.

        Returns:
            base_orn (ndarray): rotation matrix of base orientation.
        """
        return self._quat_to_rot(self.base_orn)

    def get_base_quaternion(self):
        """Get the robot's base orientation in quaternion.

        Returns:
            base_orn (ndarray): quaternion of base orientation [x,y,z,w].
        """
        return np.array(self.base_orn)

    def get_base_euler_rpy(self):
        """Get the robot's base orientation in euler angles.

        Returns:
            base_orn (ndarray): euler rpy of base orientation.
        """
        # Convert quaternion to euler angles using MuJoCo function
        euler = np.zeros(3)
        # MuJoCo uses w, x, y, z order for quaternions
        quat = self.base_orn
        mjc.mju_quat2Euler(euler, quat)
        return euler

    def get_base_linear_velocity(self):
        """Get the robot's base linear velocity in base frame.

        Returns:
            base_lin_vel (ndarray): base linear velocity.
        """
        return np.array(self.base_lin_vel)

    def get_base_angular_velocity(self):
        """Get the robot's base angular velocity in base frame.

        Returns:
            base_ang_vel (ndarray): base angular velocity.
        """
        return np.array(self.base_ang_vel)

    def get_link_pose(self, name):
        """Get the link frame pose in world frame.

        Returns:
            frame_pose (ndarray): frame pose.
        """
        try:
            body_id = mjc.mj_name2id(self.model, mjc.mjtObj.mjOBJ_BODY, name)
            # Get position from body's center of mass
            pos = self.data.xpos[body_id].copy()
            # Get orientation as quaternion from rotation matrix
            rot_mat = self.data.xmat[body_id].copy().reshape(3,3)
            orn_quat = self._rot_to_quat(rot_mat)
            frame_pose = np.concatenate((pos, orn_quat))
            return frame_pose
        except:
            return np.zeros(7)

    def get_link_velocity(self, name):
        """Get the link frame velocity in world frame.

        Returns:
            frame_velocity (ndarray): frame velocity.
        """
        try:
            body_id = mjc.mj_name2id(self.model, mjc.mjtObj.mjOBJ_BODY, name)
            # Get the velocity of the body
            lin_vel = self.data.cvel[body_id, 3:6].copy()  # Linear velocity
            ang_vel = self.data.cvel[body_id, 0:3].copy()  # Angular velocity
            frame_velocity = np.concatenate((lin_vel, ang_vel))
            return frame_velocity
        except:
            return np.zeros(6)
    
    def get_limb_contact_states(self):
        """Get robot's limb contact situation with the environment.

        Returns:
            A list of 4 booleans. The i-th boolean is True if limb i is in contact
            with the environment.
        """
        contact_states = {limb_name: False for limb_name in self.limb_names}

        for endeff_name in self.limb_endeff_names:
            try:
                endeff_id = mjc.mj_name2id(self.model, mjc.mjtObj.mjOBJ_GEOM, endeff_name)
                
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    if contact.geom1 == endeff_id or contact.geom2 == endeff_id:
                        limb_name = self.endeff_name_limb_name_map[endeff_name]
                        contact_states[limb_name] = True
                        break
            except:
                pass

        return [contact_states[limb_name] for limb_name in self.limb_names]

    def apply_joint_torques(self, names, torques):
        """Set the desired torques to the joints.

        Args:
            names: The joint names.
            torques: The desired joint torques.
        """
        for i, name in enumerate(names):
            try:
                ctrl_idx = self.joint_names.index(name)
                if ctrl_idx < len(self.data.ctrl):
                    self.data.ctrl[ctrl_idx] = torques[i]
            except:
                pass