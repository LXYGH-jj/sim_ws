"""
@file robot_wrapper.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

import numpy as np
import pinocchio as pin
from pinocchio.utils import zero

class RobotWrapper: 

    def __init__(self, setting):
        self.urdf_filename = setting.urdf_filename
        self.joint_names = setting.joint_names
        self.leg_endeff_names = setting.leg_endeff_names
        self.joint_init_pos = setting.joint_init_pos
        self.joint_init_vel = setting.joint_init_vel

        self.model = pin.buildModelFromUrdf(
            self.urdf_filename, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv

        self.pin_joint_name_id_map = {}
        self.pin_joint_id_name_map = {}
        for ni, name in enumerate(self.joint_names):
            ji = self.model.getJointId(name)
            self.pin_joint_id_name_map[ji] = name
            self.pin_joint_name_id_map[name] = ji

        self.q = zero(self.nq)
        self.v = zero(self.nv)

        # obtain real base height
        self.q[0:7] = np.array([0., 0., 0., 0., 0., 0., 1.]) # [px,py,pz,qx,qy,qz,qw]
        self.q[7:] = self.joint_init_pos
        self.v[0:6] = np.array([0., 0., 0., 0., 0., 0.]) # [vx,vy,vz,wx,wy,wz]
        self.v[6:] = self.joint_init_vel
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        heights = []
        for idx, name in enumerate(self.leg_endeff_names):
            heights.append(self.get_frame_pose(name).translation[2])
        self.q[2] = - sum(heights)/len(heights)
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)

    def set_state(self, base_pos, base_vel, joint_pos, joint_vel):
        """Set the robot to the desired states. Note that the base velocities
        are expressed in the base frame.
        
        Args:
            base_pos (ndarray): Desired base posture.
            base_vel (ndarray): Desired base velocity.
            joint_pos (ndarray): Desired joint positions.
            joint_vel (ndarray): Desired joint velocities.
        """
        self.q[0:7] = base_pos
        self.v[0:6] = base_vel
        for ji, jn in enumerate(self.joint_names):
            self.q[5 + self.pin_joint_name_id_map[jn]] = joint_pos[ji]
            self.v[4 + self.pin_joint_name_id_map[jn]] = joint_vel[ji]

        # Computes efficiently all the terms needed for dynamic simulation. It is 
        # equivalent to the call at the same time to: 
        #     - pinocchio::forwardKinematics
        #     - pinocchio::crba
        #     - pinocchio::nonLinearEffects
        #     - pinocchio::computeJointJacobians
        #     - pinocchio::centerOfMass
        #     - pinocchio::jacobianCenterOfMass
        #     - pinocchio::ccrba
        #     - pinocchio::computeKineticEnergy
        #     - pinocchio::computePotentialEnergy
        #     - pinocchio::computeGeneralizedGravity
        #
        pin.computeAllTerms(self.model, self.data, self.q, self.v)
        # updates the position of each frame to store in data.oMf
        pin.updateFramePlacements(self.model, self.data)

    def get_state(self):
        """Get the robot's position and velocity.

        Returns:
            pos (ndarray): Robot positions.
            vel (ndarray): Robot velocities.
        """
        return self.q, self.v

    def integrate(self, cur_q, v):
        """Integrate a configuration vector for a tangent vector during one unit time.

        Args:
            v (ndarray): Tangent vector.

        Return:
            q (ndarray): Updated configuration vector.
        """
        return pin.integrate(self.model, cur_q, v)

    def get_inertia_matrix(self):
        """Get the joint-space inertia matrix."""
        return self.data.M

    def get_nonlinear_effects(self):
        """Get nonlinear effects corresponding to concatenation of the coriolis, 
        centrifugal and gravitational effects.
        """
        return self.data.nle

    def get_com_position(self):
        """Vector of absolute com position.
        """
        return self.data.com[0]

    def get_com_velocity(self):
        """Vector of absolute com velocity.
        """
        return self.data.vcom[0]

    def get_com_acceleration(self):
        """Vector of absolute com accleration.
        """
        return self.data.acom[0]

    def get_com_jacobian(self):
        """Express frame jacobian in local world aligned coordinate system 
        centered on the moving part but with axes aligned with the frame of the 
        Universe.
        """
        return self.data.Jcom

    def get_frame_pose(self, name):
        """Vector of absolute frame position.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return self.data.oMf[index]

    def get_frame_velocity(self, name):
        """Express frame velocity in local coordinate system associated with the
        moving part.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameVelocity(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL)

    def get_frame_velocity_world_aligned(self, name):
        """Express frame velocity in local world aligned coordinate system 
        centered on the moving part but with axes aligned with the frame of the 
        Universe.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameVelocity(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def get_frame_acceleration(self, name):
        """Express frame acceleration in local coordinate system associated with 
        the moving part.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameAcceleration(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL)

    def get_frame_acceleration_world_aligned(self, name):
        """Express frame acceleration in local world aligned coordinate system 
        centered on the moving part but with axes aligned with the frame of the 
        Universe.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameAcceleration(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def get_frame_jacobian(self, name):
        """Express frame jacobian in local coordinate system associated with 
        the moving part.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameJacobian(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL)

    def get_frame_jacobian_world_aligned(self, name):
        """Express frame jacobian in local world aligned coordinate system 
        centered on the moving part but with axes aligned with the frame of the 
        Universe.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameJacobian(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def get_frame_jacobian_time_variation(self, name):
        """Express frame jacobian time variation in local coordinate system 
        associated with the moving part.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameJacobianTimeVariation(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL)

    def get_frame_jacobian_time_variation_world_aligned(self, name):
        """Express frame time variation in local world aligned coordinate system 
        centered on the moving part but with axes aligned with the frame of the 
        Universe.

        Args:
            name (:obj:`str`): Frame name.
        """
        if not self.model.existFrame(name):
            raise ValueError("Joint %s is not available." %name)
        if name == "universe" or name == "root_joint":
            raise ValueError("Joint %s is not available." %name)

        index = self.model.getFrameId(name)
        return pin.getFrameJacobianTimeVariation(
            self.model, self.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)