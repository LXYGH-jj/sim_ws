"""
@file command.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

class HighLevelCommand:
    """Commands from high-level planner to low-level whole body controller.
    The pose of base and end effectors has structure as follow:
    pose = {
        "pos": eef_pos, 
        "orn": eef_orn,
        "lin_vel": eef_lin_vel, 
        "ang_vel": eef_ang_vel, 
        "lin_acc": eef_lin_acc, 
        "ang_acc": eef_ang_acc
    }
    """
    def __init__(self):
        self.des_task_hierarchy = {}
        self.des_base_pose = {}
        self.des_joint_posture = {}
        self.des_stance_legs = []
        self.des_swing_legs = []
        self.des_stance_leg_endeff_wrenches = {}
        self.des_swing_leg_endeff_poses = {}
        self.des_stance_arms = []
        self.des_swing_arms = []
        self.des_stance_arm_endeff_wrenches = {}
        self.des_swing_arm_endeff_poses = {}

    def set_desired_task_hierarchy(self, des_task_hierarchy=None):
        self.des_task_hierarchy = des_task_hierarchy

    def set_desired_base_pose(self, des_base_pose=None):
        self.des_base_pose = des_base_pose

    def set_desired_joint_posture(self, des_joint_posture=None):
        self.des_joint_posture = des_joint_posture

    def set_desired_stance_legs(self, des_stance_legs=None):
        self.des_stance_legs = des_stance_legs

    def set_desired_swing_legs(self, des_swing_legs=None):
        self.des_swing_legs = des_swing_legs

    def set_desired_stance_leg_endeff_wrenches(self, des_stance_leg_endeff_wrenches=None):
        self.des_stance_leg_endeff_wrenches = des_stance_leg_endeff_wrenches

    def set_desired_swing_leg_endeff_poses(self, des_swing_leg_endeff_poses=None):
        self.des_swing_leg_endeff_poses = des_swing_leg_endeff_poses

    def set_desired_stance_arms(self, des_stance_arms=None):
        self.des_stance_arms = des_stance_arms

    def set_desired_swing_arms(self, des_swing_arms=None):
        self.des_swing_arms = des_swing_arms

    def set_desired_stance_arm_endeff_wrenches(self, des_stance_arm_endeff_wrenches=None):
        self.des_stance_arm_endeff_wrenches = des_stance_arm_endeff_wrenches

    def set_desired_swing_arm_endeff_poses(self, des_swing_arm_endeff_poses=None):
        self.des_swing_arm_endeff_poses = des_swing_arm_endeff_poses

        """@property是python的内置装饰器，用于将方法转换为只读属性"""
    @property
    def desired_task_hierarchy(self):
        return self.des_task_hierarchy

    @property
    def desired_base_pose(self):
        return self.des_base_pose

    @property
    def desired_joint_posture(self):
        return self.des_joint_posture

    @property
    def desired_stance_legs(self):
        return self.des_stance_legs

    @property
    def desired_swing_legs(self):
        return self.des_swing_legs

    @property
    def desired_stance_leg_endeff_wrenches(self):
        return self.des_stance_leg_endeff_wrenches

    @property
    def desired_swing_leg_endeff_poses(self):
        return self.des_swing_leg_endeff_poses

    @property
    def desired_stance_arms(self):
        return self.des_stance_arms

    @property
    def desired_swing_arms(self):
        return self.des_swing_arms

    @property
    def desired_stance_arm_endeff_wrenches(self):
        return self.des_stance_arm_endeff_wrenches

    @property
    def desired_swing_arm_endeff_poses(self):
        return self.des_swing_arm_endeff_poses
