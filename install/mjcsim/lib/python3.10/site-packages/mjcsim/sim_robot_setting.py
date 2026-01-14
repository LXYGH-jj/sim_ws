"""
@file sim_robot_setting.py
@package mjcsim
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

import numpy as np
# from commutils.yaml_parser import load_yaml
from commutils.yaml_parser import load_yaml

class SimRobotSetting:
    def __init__(self):
        self.xml_filename = None
        self.use_fixed_base = False
        self.base_name = None
        self.limb_names = []
        self.limb_endeff_names = []
        self.joint_names = []
        self.base_init_pos = np.concatenate((np.array([0,0,0]), np.array([0,0,0,1])))
        self.base_init_vel = np.zeros(6)
        self.joint_init_pos = np.zeros(len(self.joint_names))
        self.joint_init_vel = np.zeros(len(self.joint_names))

    def initialize(self, rootdir, cfg_file, rob_vars_yaml="sim_robot_variables"):
        configs = load_yaml(rootdir + cfg_file)
        self.xml_filename = rootdir + configs[rob_vars_yaml]["xml_filename"]
        self.use_fixed_base = configs[rob_vars_yaml]["use_fixed_base"]
        self.base_name = configs[rob_vars_yaml]["base_name"]
        self.limb_names = configs[rob_vars_yaml]["limb_names"]
        self.limb_endeff_names = configs[rob_vars_yaml]["limb_endeff_names"]
        self.joint_names = configs[rob_vars_yaml]["joint_names"]
        self.base_init_pos = np.array(configs[rob_vars_yaml]["base_init_pos"])
        self.base_init_vel = np.array(configs[rob_vars_yaml]["base_init_vel"])
        self.joint_init_pos = np.array(configs[rob_vars_yaml]["joint_init_pos"])
        self.joint_init_vel = np.array(configs[rob_vars_yaml]["joint_init_vel"])