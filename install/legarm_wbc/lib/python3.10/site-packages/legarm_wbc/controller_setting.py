"""
@file controller_setting.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

import numpy as np
from commutils.yaml_parser import load_yaml

class ControllerSetting:

    def __init__(self):
        self.urdf_filename = None
        self.base_name = []
        self.leg_names = []
        self.arm_names = []
        self.limb_names = []
        self.leg_endeff_names = []
        self.arm_endeff_names = []
        self.limb_endeff_names = []
        self.joint_names = []
        self.torque_limits = np.zeros(len(self.joint_names))
        self.torque_factor = 0.
        self.fric_coef = 1.
        self.timestep = 0.002
        self.joint_kp = np.zeros(len(self.joint_names))
        self.joint_kd = np.zeros(len(self.joint_names))
        self.kp_com_pos = np.zeros(3)
        self.kd_com_pos = np.zeros(3)
        self.kp_base_pos = np.zeros(3)
        self.kd_base_pos = np.zeros(3)
        self.kp_base_orn = np.zeros(3)
        self.kd_base_orn = np.zeros(3)
        self.kp_nom_pos = np.zeros(len(self.joint_names))
        self.kd_nom_pos = np.zeros(len(self.joint_names))
        self.kp_leg_endeff_pos = np.zeros(3)
        self.kd_leg_endeff_pos = np.zeros(3)
        self.kp_leg_endeff_orn = np.zeros(3)
        self.kd_leg_endeff_orn = np.zeros(3)
        self.kp_arm_endeff_pos = np.zeros(3)
        self.kd_arm_endeff_pos = np.zeros(3)
        self.kp_arm_endeff_orn = np.zeros(3)
        self.kd_arm_endeff_orn = np.zeros(3)

        self.joint_init_pos = np.zeros(len(self.joint_names))
        self.joint_init_vel = np.zeros(len(self.joint_names))

    def initialize(self, rootdir, cfg_file, ctrl_vars_yaml="controller_variables"):
        configs = load_yaml(rootdir + cfg_file)
        self.urdf_filename = rootdir + configs[ctrl_vars_yaml]["urdf_filename"]
        print("ControllerSetting: urdf_filename:", self.urdf_filename)
        self.base_name = configs[ctrl_vars_yaml]["base_name"]
        self.leg_names = configs[ctrl_vars_yaml]["leg_names"]
        self.arm_names = configs[ctrl_vars_yaml]["arm_names"]
        self.limb_names = configs[ctrl_vars_yaml]["limb_names"]
        self.leg_endeff_names = configs[ctrl_vars_yaml]["leg_endeff_names"]
        self.arm_endeff_names = configs[ctrl_vars_yaml]["arm_endeff_names"]
        self.limb_endeff_names = configs[ctrl_vars_yaml]["limb_endeff_names"]
        self.joint_names = configs[ctrl_vars_yaml]["joint_names"]
        self.torque_limits = np.array(configs[ctrl_vars_yaml]["torque_limits"])
        self.torque_factor = configs[ctrl_vars_yaml]["torque_factor"]
        self.fric_coef = configs[ctrl_vars_yaml]["fric_coef"]
        self.timestep = configs[ctrl_vars_yaml]["timestep"]
        self.joint_kp = np.array(configs[ctrl_vars_yaml]["joint_kp"])
        self.joint_kd = np.array(configs[ctrl_vars_yaml]["joint_kd"])

        self.kp_base_pos = np.array(configs[ctrl_vars_yaml]["kp_base_pos"])
        self.kd_base_pos = np.array(configs[ctrl_vars_yaml]["kd_base_pos"])
        self.kp_base_orn = np.array(configs[ctrl_vars_yaml]["kp_base_orn"])
        self.kd_base_orn = np.array(configs[ctrl_vars_yaml]["kd_base_orn"])
        self.kp_nom_pos = np.array(configs[ctrl_vars_yaml]["kp_nom_pos"])
        self.kd_nom_pos = np.array(configs[ctrl_vars_yaml]["kd_nom_pos"])
        self.kp_leg_endeff_pos = np.array(configs[ctrl_vars_yaml]["kp_leg_endeff_pos"])
        self.kd_leg_endeff_pos = np.array(configs[ctrl_vars_yaml]["kd_leg_endeff_pos"])
        self.kp_leg_endeff_orn = np.array(configs[ctrl_vars_yaml]["kp_leg_endeff_orn"])
        self.kd_leg_endeff_orn = np.array(configs[ctrl_vars_yaml]["kd_leg_endeff_orn"])
        self.kp_arm_endeff_pos = np.array(configs[ctrl_vars_yaml]["kp_arm_endeff_pos"])
        self.kd_arm_endeff_pos = np.array(configs[ctrl_vars_yaml]["kd_arm_endeff_pos"])
        self.kp_arm_endeff_orn = np.array(configs[ctrl_vars_yaml]["kp_arm_endeff_orn"])
        self.kd_arm_endeff_orn = np.array(configs[ctrl_vars_yaml]["kd_arm_endeff_orn"])

        self.joint_init_pos = np.array(configs[ctrl_vars_yaml]["joint_init_pos"])
        self.joint_init_vel = np.array(configs[ctrl_vars_yaml]["joint_init_vel"])
