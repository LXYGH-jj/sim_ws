"""
@file whole_body_controller.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

import numpy as np
from scipy.linalg import block_diag
from legarm_wbc.geometry import positionPD, rotationPD
from legarm_wbc.hqp_data import HQPData
from legarm_wbc.hqp_solver import HQPSolver

class WholeBodyController:
    """The hierarchical inverse dynamics controller serves as a whole body 
    controller, which is responsible for generating the desired actuator commands 
    using the hierarchical optimization algorithm.
    """
    def __init__(self, setting):
        self.base_name = setting.base_name
        self.leg_names = setting.leg_names
        self.arm_names = setting.arm_names
        self.limb_names = setting.limb_names
        self.leg_endeff_names = setting.leg_endeff_names
        self.arm_endeff_names = setting.arm_endeff_names
        self.limb_endeff_names = setting.limb_endeff_names
        self.joint_names = setting.joint_names
        self.torque_limits = setting.torque_limits
        self.torque_factor = setting.torque_factor
        self.fric_coef = setting.fric_coef
        self.timestep = setting.timestep
        self.joint_kp = setting.joint_kp
        self.joint_kd = setting.joint_kd
        self.kp_com_pos = setting.kp_com_pos
        self.kd_com_pos = setting.kd_com_pos
        self.kp_base_pos = setting.kp_base_pos
        self.kd_base_pos = setting.kd_base_pos
        self.kp_base_orn = setting.kp_base_orn
        self.kd_base_orn = setting.kd_base_orn
        self.kp_nom_pos = setting.kp_nom_pos
        self.kd_nom_pos = setting.kd_nom_pos
        self.kp_leg_endeff_pos = setting.kp_leg_endeff_pos
        self.kd_leg_endeff_pos = setting.kd_leg_endeff_pos
        self.kp_leg_endeff_orn = setting.kp_leg_endeff_orn
        self.kd_leg_endeff_orn = setting.kd_leg_endeff_orn
        self.kp_arm_endeff_pos = setting.kp_arm_endeff_pos
        self.kd_arm_endeff_pos = setting.kd_arm_endeff_pos
        self.kp_arm_endeff_orn = setting.kp_arm_endeff_orn
        self.kd_arm_endeff_orn = setting.kd_arm_endeff_orn

        assert len(self.leg_names) == len(self.leg_endeff_names)
        assert len(self.arm_names) == len(self.arm_endeff_names)
        assert len(self.limb_names) == len(self.limb_endeff_names)

        self.leg_name_endeff_name_map = {}
        for idx in range(len(self.leg_names)):
            self.leg_name_endeff_name_map[
                self.leg_names[idx]] = self.leg_endeff_names[idx]

        self.arm_name_endeff_name_map = {}
        for idx in range(len(self.arm_names)):
            self.arm_name_endeff_name_map[
                self.arm_names[idx]] = self.arm_endeff_names[idx]

        self.limb_name_endeff_name_map = {}
        for idx in range(len(self.limb_names)):
            self.limb_name_endeff_name_map[
                self.limb_names[idx]] = self.limb_endeff_names[idx]

        self.des_q = np.zeros(7+len(self.joint_names))
        self.des_v = np.zeros(6+len(self.joint_names))
        self.des_dv = np.zeros(6+len(self.joint_names))
        self.des_lambdas = None

        self.des_tau = np.zeros(6+len(self.joint_names))

        self.des_leg_endeff_forces = {} # ground reaction forces
        for leg_name in self.leg_names:
            endeff_name = self.leg_name_endeff_name_map[leg_name]
            self.des_leg_endeff_forces[endeff_name] = np.zeros(3)

        self.hqp_data = HQPData()
        self.hqp_solver = HQPSolver()

    def get_contact_jacobian(self, robot_wrapper, stance_legs):
        J_contact = []
        for stance_leg in stance_legs:
            endeff_name = self.leg_name_endeff_name_map[stance_leg]
            J_contact.append(
                robot_wrapper.get_frame_jacobian_world_aligned(endeff_name)[0:3])
        if len(J_contact):
            J_contact = np.vstack(J_contact)
            """使用 np.vstack 将所有雅可比矩阵垂直堆叠成一个大矩阵"""
        else:
            J_contact = np.empty((0, 6+len(self.joint_names)))
        return J_contact

    def get_dynamic_consistency_task(self, inertia_mat, nle, j_cont):
        base_dyn_mat = np.hstack((inertia_mat[:6, :], -j_cont.T[:6, :]))  # 6*(N+12)
        base_dyn_vec = - nle[:6]  # 6*1
        return (base_dyn_mat, base_dyn_vec)

    def get_friction_cone_constraints(self, dv_dim, num_of_lambdas):
        # Friction constraints
        fric_cons = np.array([
            [ 1,  0, -self.fric_coef], 
            [-1,  0, -self.fric_coef], 
            [ 0,  1, -self.fric_coef], 
            [ 0, -1, -self.fric_coef]
        ])
        # Unilateral constraints
        unil_cons = np.array([[0, 0, -1]])

        # Lambda constraints: friction constraints + unilateral constraints
        force_cons = np.vstack((fric_cons, unil_cons))
        lambda_cons = []
        for i in range(num_of_lambdas):
            lambda_cons = block_diag(lambda_cons, force_cons)

        if len(lambda_cons):
            fric_cons_mat = np.hstack(
                (np.zeros((lambda_cons.shape[0], dv_dim)), lambda_cons))
            fric_cons_vec = np.zeros((lambda_cons.shape[0]))
        else:
            fric_cons_mat = np.empty((0, dv_dim))
            fric_cons_vec = np.empty((0))
        return (fric_cons_mat, fric_cons_vec)

    def get_torque_saturation_constraints(self, inertia_mat, nle, j_cont):
        trq_cons = np.hstack((inertia_mat[6:,:], -j_cont.T[6:,:]))
        trq_limits = self.torque_factor * self.torque_limits
        trq_cons_mat = np.vstack((-trq_cons, trq_cons))
        trq_cons_vec = np.hstack((+nle[6:]+trq_limits, -nle[6:]+trq_limits))
        return (trq_cons_mat, trq_cons_vec)

    def compute_joint_actions(self, robot_wrapper, command):
        num_of_lambdas = len(command.desired_stance_legs)

        lambda_dim = num_of_lambdas * 3
        dv_dim = robot_wrapper.nv
        x_dim = dv_dim + lambda_dim

        self.hqp_data.set_num_variables(x_dim)

        J_contact = self.get_contact_jacobian(
            robot_wrapper, command.desired_stance_legs)

        fric_cons_mat, fric_cons_vec = self.get_friction_cone_constraints(
            dv_dim, num_of_lambdas)
        self.hqp_data.add_constraint("fric_cone_cons", fric_cons_mat, fric_cons_vec)

        trq_cons_mat, trq_cons_vec = self.get_torque_saturation_constraints(
            robot_wrapper.get_inertia_matrix(), 
            robot_wrapper.get_nonlinear_effects(), 
            J_contact)
        self.hqp_data.add_constraint("trq_sat_cons", trq_cons_mat, trq_cons_vec)

        if "dynamic_consistency_task" in command.desired_task_hierarchy.keys():
            # Dynamic consistency task
            base_dyn_mat, base_dyn_vec = self.get_dynamic_consistency_task(
                robot_wrapper.get_inertia_matrix(), 
                robot_wrapper.get_nonlinear_effects(), 
                J_contact)
            if "dynamic_consistency_task" not in self.hqp_data.hierarchical_tasks.keys():
                self.hqp_data.hierarchical_tasks["dynamic_consistency_task"] = {
                    "priority":command.desired_task_hierarchy[
                        "dynamic_consistency_task"]["priority"], 
                    "weight":command.desired_task_hierarchy[
                        "dynamic_consistency_task"]["weight"], 
                    "A":base_dyn_mat, 
                    "b":base_dyn_vec}
            else:
                self.hqp_data.hierarchical_tasks["dynamic_consistency_task"][
                    "A"] = base_dyn_mat
                self.hqp_data.hierarchical_tasks["dynamic_consistency_task"][
                    "b"] = base_dyn_vec

        if "com_position_task" in command.desired_task_hierarchy.keys():
            # Com position task
            cur_com_pos = robot_wrapper.get_com_position()
            cur_com_vel = robot_wrapper.get_com_velocity()
            com_acc = positionPD(
                des_pos=command.desired_com_position["com"]["pos"],
                cur_pos=cur_com_pos, 
                des_vel=command.desired_com_position["com"]["lin_vel"],
                cur_vel=cur_com_vel,
                des_acc=command.desired_com_position["com"]["lin_acc"],
                kp=self.kp_com_pos, 
                kd=self.kd_com_pos)
            com_pos_mat = np.hstack(
                (robot_wrapper.get_com_jacobian(), 
                np.zeros((3, lambda_dim)))
            )
            com_pos_vec = (
                com_acc - robot_wrapper.get_com_acceleration()
            )
            if "com_position_task" not in self.hqp_data.hierarchical_tasks.keys():
                self.hqp_data.hierarchical_tasks["com_position_task"] = {
                    "priority":command.desired_task_hierarchy[
                        "com_position_task"]["priority"], 
                    "weight":command.desired_task_hierarchy[
                        "com_position_task"]["weight"], 
                    "A":com_pos_mat, 
                    "b":com_pos_vec}
            else:
                self.hqp_data.hierarchical_tasks["com_position_task"][
                    "A"] = com_pos_mat
                self.hqp_data.hierarchical_tasks["com_position_task"][
                    "b"] = com_pos_vec

        if "base_position_task" in command.desired_task_hierarchy.keys():
            # Base position task
            cur_base_pos = robot_wrapper.get_frame_pose(
                self.base_name).translation
            cur_base_lin_vel = robot_wrapper.get_frame_velocity_world_aligned(
                self.base_name).linear
            base_lin_acc = positionPD(
                des_pos=command.desired_base_pose[self.base_name]["pos"], 
                cur_pos=cur_base_pos, 
                des_vel=command.desired_base_pose[self.base_name]["lin_vel"], 
                cur_vel=cur_base_lin_vel, 
                des_acc=command.desired_base_pose[self.base_name]["lin_acc"], 
                kp=self.kp_base_pos, 
                kd=self.kd_base_pos)
            base_pos_mat = np.hstack(
                (robot_wrapper.get_frame_jacobian_world_aligned(self.base_name)[0:3], 
                 np.zeros((3, lambda_dim)))
            )
            base_pos_vec = (
                base_lin_acc - 
                robot_wrapper.get_frame_jacobian_time_variation_world_aligned(
                    self.base_name)[0:3].dot(robot_wrapper.v)
            )
            if "base_position_task" not in self.hqp_data.hierarchical_tasks.keys():
                self.hqp_data.hierarchical_tasks["base_position_task"] = {
                    "priority":command.desired_task_hierarchy[
                        "base_position_task"]["priority"], 
                    "weight":command.desired_task_hierarchy[
                        "base_position_task"]["weight"], 
                    "A":base_pos_mat, 
                    "b":base_pos_vec}
            else:
                self.hqp_data.hierarchical_tasks["base_position_task"][
                    "A"] = base_pos_mat
                self.hqp_data.hierarchical_tasks["base_position_task"][
                    "b"] = base_pos_vec
 
        if "base_orientation_task" in command.desired_task_hierarchy.keys():
            # Base orientation task
            cur_base_orn = robot_wrapper.get_frame_pose(self.base_name).rotation
            cur_base_ang_vel = robot_wrapper.get_frame_velocity_world_aligned(
                    self.base_name).angular
            base_ang_acc = rotationPD(
                des_rot=command.desired_base_pose[self.base_name]["orn"], 
                cur_rot=cur_base_orn,
                des_omega=command.desired_base_pose[self.base_name]["ang_vel"], 
                cur_omega=cur_base_ang_vel,  
                des_omega_dot=command.desired_base_pose[self.base_name]["ang_acc"], 
                kp=self.kp_base_orn, 
                kd=self.kd_base_orn)
            base_orn_mat = np.hstack(
                (robot_wrapper.get_frame_jacobian_world_aligned(self.base_name)[3:6], 
                 np.zeros((3, lambda_dim)))
            )
            base_orn_vec = (
                base_ang_acc - 
                robot_wrapper.get_frame_jacobian_time_variation_world_aligned(
                    self.base_name)[3:6].dot(robot_wrapper.v)
            )
            if "base_orientation_task" not in self.hqp_data.hierarchical_tasks.keys():
                self.hqp_data.hierarchical_tasks["base_orientation_task"] = {
                    "priority":command.desired_task_hierarchy[
                        "base_orientation_task"]["priority"], 
                    "weight":command.desired_task_hierarchy[
                        "base_orientation_task"]["weight"], 
                    "A":base_orn_mat, 
                    "b":base_orn_vec}
            else:
                self.hqp_data.hierarchical_tasks["base_orientation_task"][
                    "A"] = base_orn_mat
                self.hqp_data.hierarchical_tasks["base_orientation_task"][
                    "b"] = base_orn_vec

        if "contact_motion_task" in command.desired_task_hierarchy.keys():
            # Contact motion task
            contact_motion_mat_all, contact_motion_vec_all = [], []
            for stance_leg in command.desired_stance_legs:
                endeff_name = self.leg_name_endeff_name_map[stance_leg]
                contact_motion_mat = np.hstack(
                    (robot_wrapper.get_frame_jacobian_world_aligned(endeff_name)[0:3], 
                     np.zeros((3, lambda_dim)))
                )
                contact_motion_vec = (
                    - robot_wrapper.get_frame_jacobian_time_variation_world_aligned(
                        endeff_name)[0:3].dot(robot_wrapper.v)
                )
                contact_motion_mat_all.append(contact_motion_mat)
                contact_motion_vec_all.append(contact_motion_vec)
            if (len(contact_motion_mat_all) > 0 and len(contact_motion_vec_all) > 0):
                if "contact_motion_task" not in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks["contact_motion_task"] = {
                        "priority":command.desired_task_hierarchy[
                            "contact_motion_task"]["priority"],
                        "weight":command.desired_task_hierarchy[
                            "contact_motion_task"]["weight"], 
                        "A":np.vstack(contact_motion_mat_all), 
                        "b":np.hstack(contact_motion_vec_all)}
                else:
                    self.hqp_data.hierarchical_tasks["contact_motion_task"][
                        "A"] = np.vstack(contact_motion_mat_all)
                    self.hqp_data.hierarchical_tasks["contact_motion_task"][
                        "b"] = np.hstack(contact_motion_vec_all)
            else:
                if "contact_motion_task" in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks.pop("contact_motion_task")

        if "contact_force_task" in command.desired_task_hierarchy.keys():
            # Contact force task
            contact_force_mat_all, contact_force_vec_all = [], []
            for i, stance_leg in enumerate(command.desired_stance_legs):
                endeff_name = self.leg_name_endeff_name_map[stance_leg]
                contact_force_mat = np.hstack((np.zeros((3, dv_dim)), np.zeros((3, lambda_dim))))
                contact_force_mat[:3, dv_dim + 3*i: dv_dim+3*(i+1)] = np.identity(3)
                contact_force_vec = command.desired_stance_leg_endeff_wrenches[endeff_name]["force"]
                contact_force_mat_all.append(contact_force_mat)
                contact_force_vec_all.append(contact_force_vec)
            if (len(contact_force_mat_all) > 0 and len(contact_force_vec_all) > 0):
                if "contact_force_task" not in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks["contact_force_task"] = {
                        "priority":command.desired_task_hierarchy[
                            "contact_force_task"]["priority"],
                        "weight":command.desired_task_hierarchy[
                            "contact_force_task"]["weight"], 
                        "A":np.vstack(contact_force_mat_all), 
                        "b":np.hstack(contact_force_vec_all)}
                else:
                    self.hqp_data.hierarchical_tasks["contact_force_task"][
                        "A"] = np.vstack(contact_force_mat_all)
                    self.hqp_data.hierarchical_tasks["contact_force_task"][
                        "b"] = np.hstack(contact_force_vec_all)
            else:
                if "contact_force_task" in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks.pop("contact_force_task")

        if "swing_leg_endeff_position_task" in command.desired_task_hierarchy.keys():
            # Swing leg endeff position task
            swing_leg_endeff_mat_all, swing_leg_endeff_vec_all = [], []
            for swing_leg in command.desired_swing_legs:
                endeff_name = self.leg_name_endeff_name_map[swing_leg]
                cur_leg_endeff_pos = robot_wrapper.get_frame_pose(
                    endeff_name).translation
                cur_leg_endeff_lin_vel = robot_wrapper.get_frame_velocity_world_aligned(
                    endeff_name).linear
                swing_leg_endeff_lin_acc = positionPD(
                    des_pos=command.desired_swing_leg_endeff_poses[endeff_name]["pos"], 
                    cur_pos=cur_leg_endeff_pos, 
                    des_vel=command.desired_swing_leg_endeff_poses[endeff_name]["lin_vel"], 
                    cur_vel=cur_leg_endeff_lin_vel, 
                    des_acc=command.desired_swing_leg_endeff_poses[endeff_name]["lin_acc"], 
                    kp=self.kp_leg_endeff_pos, 
                    kd=self.kd_leg_endeff_pos)
                swing_leg_endeff_mat = np.hstack(
                    (robot_wrapper.get_frame_jacobian_world_aligned(endeff_name)[0:3],
                     np.zeros((3, lambda_dim)))
                )
                swing_leg_endeff_vec = (
                    swing_leg_endeff_lin_acc - 
                    robot_wrapper.get_frame_jacobian_world_aligned(
                        endeff_name)[0:3].dot(robot_wrapper.v)
                )
                swing_leg_endeff_mat_all.append(swing_leg_endeff_mat)
                swing_leg_endeff_vec_all.append(swing_leg_endeff_vec)

            if (len(swing_leg_endeff_mat_all) > 0 and len(swing_leg_endeff_vec_all) > 0):
                if "swing_leg_endeff_position_task" not in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks["swing_leg_endeff_position_task"] = {
                        "priority":command.desired_task_hierarchy[
                            "swing_leg_endeff_position_task"]["priority"],
                        "weight":command.desired_task_hierarchy[
                            "swing_leg_endeff_position_task"]["weight"], 
                        "A":np.vstack(swing_leg_endeff_mat_all), 
                        "b":np.hstack(swing_leg_endeff_vec_all)}
                else:
                    self.hqp_data.hierarchical_tasks["swing_leg_endeff_position_task"][
                        "A"] = np.vstack(swing_leg_endeff_mat_all)
                    self.hqp_data.hierarchical_tasks["swing_leg_endeff_position_task"][
                        "b"] = np.hstack(swing_leg_endeff_vec_all)
            else:
                if "swing_leg_endeff_position_task" in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks.pop("swing_leg_endeff_position_task")

        if "swing_arm_endeff_position_task" in command.desired_task_hierarchy.keys():
            # Swing arm endeff position task
            swing_arm_endeff_mat_all, swing_arm_endeff_vec_all = [], []
            for swing_arm in command.desired_swing_arms:
                endeff_name = self.limb_name_endeff_name_map[swing_arm]
                cur_arm_endeff_pos = robot_wrapper.get_frame_pose(
                    endeff_name).translation
                cur_arm_endeff_lin_vel = robot_wrapper.get_frame_velocity_world_aligned(
                    endeff_name).linear
                swing_arm_endeff_lin_acc = positionPD(
                    des_pos=command.desired_swing_arm_endeff_poses[endeff_name]["pos"], 
                    cur_pos=cur_arm_endeff_pos, 
                    des_vel=command.desired_swing_arm_endeff_poses[endeff_name]["lin_vel"], 
                    cur_vel=cur_arm_endeff_lin_vel, 
                    des_acc=command.desired_swing_arm_endeff_poses[endeff_name]["lin_acc"], 
                    kp=self.kp_arm_endeff_pos, 
                    kd=self.kd_arm_endeff_pos)
                swing_arm_endeff_mat = np.hstack(
                    (robot_wrapper.get_frame_jacobian_world_aligned(endeff_name)[0:3], 
                     np.zeros((3, lambda_dim)))
                )
                swing_arm_endeff_vec = (
                    swing_arm_endeff_lin_acc - 
                    robot_wrapper.get_frame_jacobian_time_variation_world_aligned(
                        endeff_name)[0:3].dot(robot_wrapper.v)
                )
                swing_arm_endeff_mat_all.append(swing_arm_endeff_mat)
                swing_arm_endeff_vec_all.append(swing_arm_endeff_vec)
            if (len(swing_arm_endeff_mat_all) > 0 and len(swing_arm_endeff_vec_all) > 0):
                if "swing_arm_endeff_position_task" not in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks["swing_arm_endeff_position_task"] = {
                        "priority":command.desired_task_hierarchy[
                            "swing_arm_endeff_position_task"]["priority"],
                        "weight":command.desired_task_hierarchy[
                            "swing_arm_endeff_position_task"]["weight"], 
                        "A": np.vstack(swing_arm_endeff_mat_all), 
                        "b": np.hstack(swing_arm_endeff_vec_all)}
                else:
                    self.hqp_data.hierarchical_tasks["swing_arm_endeff_position_task"][
                        "A"] = np.vstack(swing_arm_endeff_mat_all)
                    self.hqp_data.hierarchical_tasks["swing_arm_endeff_position_task"][
                        "b"] = np.hstack(swing_arm_endeff_vec_all)
            else:
                if "swing_arm_endeff_position_task" in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks.pop("swing_arm_endeff_position_task")

        if "swing_arm_endeff_orientation_task" in command.desired_task_hierarchy.keys():
            # Swing arm endeff orientation task
            swing_arm_endeff_mat_all, swing_arm_endeff_vec_all = [], []
            for swing_arm in command.desired_swing_arms:
                endeff_name = self.limb_name_endeff_name_map[swing_arm]
                cur_arm_endeff_orn = robot_wrapper.get_frame_pose(
                    endeff_name).rotation
                cur_arm_endeff_ang_vel = robot_wrapper.get_frame_velocity_world_aligned(
                    endeff_name).angular
                swing_arm_endeff_ang_acc = rotationPD(
                    des_rot=command.desired_swing_arm_endeff_poses[endeff_name]["orn"], 
                    cur_rot=cur_arm_endeff_orn, 
                    des_omega=command.desired_swing_arm_endeff_poses[endeff_name]["ang_vel"], 
                    cur_omega=cur_arm_endeff_ang_vel, 
                    des_omega_dot=command.desired_swing_arm_endeff_poses[endeff_name]["ang_acc"], 
                    kp=self.kp_arm_endeff_orn, 
                    kd=self.kd_arm_endeff_orn)
                swing_arm_endeff_mat = np.hstack(
                    (robot_wrapper.get_frame_jacobian_world_aligned(endeff_name)[3:6], 
                     np.zeros((3, lambda_dim)))
                )
                swing_arm_endeff_vec = (
                    swing_arm_endeff_ang_acc - 
                    robot_wrapper.get_frame_jacobian_time_variation_world_aligned(
                        endeff_name)[3:6].dot(robot_wrapper.v)
                )
                swing_arm_endeff_mat_all.append(swing_arm_endeff_mat)
                swing_arm_endeff_vec_all.append(swing_arm_endeff_vec)
            if (len(swing_arm_endeff_mat_all) > 0 and len(swing_arm_endeff_vec_all) > 0):
                if "swing_arm_endeff_orientation_task" not in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks["swing_arm_endeff_orientation_task"] = {
                        "priority":command.desired_task_hierarchy[
                            "swing_arm_endeff_orientation_task"]["priority"],
                        "weight":command.desired_task_hierarchy[
                            "swing_arm_endeff_orientation_task"]["weight"], 
                        "A":np.vstack(swing_arm_endeff_mat_all), 
                        "b":np.hstack(swing_arm_endeff_vec_all)}
                else:
                    self.hqp_data.hierarchical_tasks["swing_arm_endeff_orientation_task"][
                        "A"] = np.vstack(swing_arm_endeff_mat_all)
                    self.hqp_data.hierarchical_tasks["swing_arm_endeff_orientation_task"][
                        "b"] = np.hstack(swing_arm_endeff_vec_all)
            else:
                if "swing_arm_endeff_orientation_task" in self.hqp_data.hierarchical_tasks.keys():
                    self.hqp_data.hierarchical_tasks.pop("swing_arm_endeff_orientation_task")

        if "minimum_motion_task" in command.desired_task_hierarchy.keys():
            # Minimum motion task
            cur_joint_pos = robot_wrapper.get_state()[0][7:]
            cur_joint_vel = robot_wrapper.get_state()[1][6:]
            joint_acc = positionPD(
                des_pos=command.desired_joint_posture["joint"]["pos"],
                cur_pos=cur_joint_pos, 
                des_vel=command.desired_joint_posture["joint"]["lin_vel"], 
                cur_vel=cur_joint_vel, 
                des_acc=command.desired_joint_posture["joint"]["lin_acc"], 
                kp=self.kp_nom_pos, 
                kd=self.kd_nom_pos
            )
            joint_pos_mat = np.hstack((np.identity(dv_dim)[6:,], np.zeros((dv_dim, lambda_dim))[6:]))
            joint_pos_vec = joint_acc
            if "minimum_motion_task" not in self.hqp_data.hierarchical_tasks.keys():
                self.hqp_data.hierarchical_tasks["minimum_motion_task"] = {
                    "priority":command.desired_task_hierarchy[
                        "minimum_motion_task"]["priority"],
                    "weight":command.desired_task_hierarchy[
                        "minimum_motion_task"]["weight"], 
                    "A":joint_pos_mat, 
                    "b":joint_pos_vec}
            else:
                self.hqp_data.hierarchical_tasks["minimum_motion_task"][
                    "A"] = joint_pos_mat
                self.hqp_data.hierarchical_tasks["minimum_motion_task"][
                    "b"] = joint_pos_vec

        x = self.hqp_solver.solve(self.hqp_data)
        dv, lambdas = x[:dv_dim], x[dv_dim:]
        self.des_dv = dv
        self.des_lambdas = lambdas

        for idx, stance_leg in enumerate(command.desired_stance_legs):
            endeff_name = self.leg_name_endeff_name_map[stance_leg]
            self.des_leg_endeff_forces[endeff_name] = lambdas[3*idx:3*idx+3]

        for i, leg_name in enumerate(self.leg_names):
            if leg_name not in command.desired_stance_legs:
                endeff_name = self.leg_name_endeff_name_map[leg_name]
                self.des_leg_endeff_forces[endeff_name] = np.zeros(3)

        ###################
        # Inverse Dynamic #
        ###################
        tau = (robot_wrapper.get_inertia_matrix().dot(dv) + 
               robot_wrapper.get_nonlinear_effects() - J_contact.T.dot(lambdas))

        self.des_tau = tau

        cur_q, cur_v = robot_wrapper.get_state()

        v_mean = cur_v + 0.5*dv*self.timestep
        self.des_v = cur_v + dv*self.timestep
        self.des_q = robot_wrapper.integrate(cur_q, v_mean*self.timestep)
        des_actions = []
        for idx, jn in enumerate(self.joint_names):
            ji = robot_wrapper.pin_joint_name_id_map[jn]
            des_actions[idx*5:idx*5+4] = [
                self.des_q[5 + ji], self.joint_kp[idx], self.des_v[4 + ji], self.joint_kd[idx], tau[4 + ji]
            ]

        return des_actions
