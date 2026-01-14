"""
@file demo_b2z1_switsh.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01

Simple demo showing how the simulation setup works and enable robot grasping.
"""

import os
import sys
import yaml
import inspect
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from commutils.yaml_parser import load_yaml
from mjcsim.env import MujocoEnvWithGround
from mjcsim.sim_robot_setting import SimRobotSetting
from mjcsim.sim_robot_interface import SimRobotInterface

from legarm_wbc.controller_setting import ControllerSetting
from legarm_wbc.robot_wrapper import RobotWrapper
from legarm_wbc.whole_body_controller import WholeBodyController
from legarm_wbc.command import HighLevelCommand


# absolute directory of this package
rootdir = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))))
TIME_STEP = 0.002  # 500 Hz
MAX_TIME_SECS = 30  # maximum time to run the robot.

class PDController:
    """A simple PD controller, used for the initial stabilization phase"""
    def __init__(self, joint_kp, joint_kd, joint_names):
        self.joint_kp = joint_kp
        self.joint_kd = joint_kd
        self.joint_names = joint_names
        self.desired_positions = None
        
    def set_desired_positions(self, desired_positions):
        self.desired_positions = desired_positions
        
    def compute_control(self, current_positions, current_velocities):
        if self.desired_positions is None:
            raise ValueError("Desired positions not set!")
            
        # PD: tau = kp*(q_des - q) + kd*(dq_des - dq)
        desired_velocities = np.zeros_like(current_velocities)
        
        position_errors = self.desired_positions - current_positions
        velocity_errors = desired_velocities - current_velocities
        
        torques = self.joint_kp * position_errors + self.joint_kd * velocity_errors
        
        actions = []
        for idx, jn in enumerate(self.joint_names):
            # Format: [desired_pos, kp, desired_vel, kd, torque]
            actions.extend([
                self.desired_positions[idx],  
                self.joint_kp[idx],           
                0.0,                         
                self.joint_kd[idx],           
                torques[idx]                  
            ])
            
        return actions

def main(argv):
    if len(argv) == 1:
        cfg_file = argv[0]
    else:
        raise RuntimeError("Usage: python3 ./demo.py /<config file within root folder>")
    
    with open(rootdir + cfg_file, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    print("model_filename: ", cfgs['sim_robot_variables']['xml_filename'])

    configs = load_yaml(rootdir + cfg_file)
    timestep = configs["timestep"]
    scale = configs["scale"]
    duration = configs["duration"]
    scaled_duration = duration * scale
    
    pd_control_duration = 5.0  
    use_pd_control = True      
    
    env = MujocoEnvWithGround(dt=timestep)

    
    sim_setting = SimRobotSetting()
    sim_setting.initialize(rootdir, cfg_file)
    sim_robot = SimRobotInterface(sim_setting)
    env.add_robot(sim_robot)

    
    ctrl_setting = ControllerSetting()
    ctrl_setting.initialize(rootdir, cfg_file)
    robwrapper = RobotWrapper(ctrl_setting)
    controller = WholeBodyController(ctrl_setting)

    
    pd_controller = PDController(
        joint_kp=np.array(configs["controller_variables"]["joint_kp"]),
        joint_kd=np.array(configs["controller_variables"]["joint_kd"]),
        joint_names=configs["controller_variables"]["joint_names"]
    )

    
    command = HighLevelCommand()
    des_task_hierarchy = configs["planner_variables"]["task_hierarchy"]
    des_base_pose = {}
    des_base_pose[configs["planner_variables"]["base_name"]] = {
        "pos": np.array(configs["planner_variables"]["base_des_pos"][0:3]), 
        "orn": R.from_quat(configs["planner_variables"]["base_des_pos"][3:7]).as_matrix(), 
        "lin_vel": np.array(configs["planner_variables"]["base_des_vel"][0:3]), 
        "ang_vel": np.array(configs["planner_variables"]["base_des_vel"][3:6]), 
        "lin_acc": np.zeros(3), 
        "ang_acc": np.zeros(3)
    }
    des_joint_posture = {}
    des_joint_posture["joint"] = {
        "pos": np.array(configs["planner_variables"]["joint_des_pos"]),
        "lin_vel": np.array(configs["planner_variables"]["joint_des_vel"]),
        "lin_acc": np.zeros(len(np.array(configs["planner_variables"]["joint_des_vel"]))),
    }
    des_stance_legs = configs["planner_variables"]["leg_names"]
    des_swing_legs = []
    des_stance_arms = []
    des_swing_arms = configs["planner_variables"]["arm_names"]
    des_swing_arm_endeff_poses = {}
    des_swing_arm_endeff_poses["link06"] = {
        "pos": np.array(configs["planner_variables"]["arm_des_pos"][0:3]), 
        "orn": R.from_quat(configs["planner_variables"]["arm_des_pos"][3:7]).as_matrix(), 
        "lin_vel": np.array(configs["planner_variables"]["arm_des_vel"][0:3]), 
        "ang_vel": np.array(configs["planner_variables"]["arm_des_vel"][3:6]), 
        "lin_acc": np.zeros(3), 
        "ang_acc": np.zeros(3)
    }
    command.set_desired_task_hierarchy(des_task_hierarchy)
    command.set_desired_base_pose(des_base_pose)
    command.set_desired_joint_posture(des_joint_posture)
    command.set_desired_stance_legs(des_stance_legs)
    command.set_desired_swing_legs(des_swing_legs)
    command.set_desired_stance_arms(des_stance_arms)
    command.set_desired_swing_arms(des_swing_arms)
    command.set_desired_swing_arm_endeff_poses(des_swing_arm_endeff_poses)

    
    pd_controller.set_desired_positions(np.array(configs["planner_variables"]["joint_des_pos"]))

    start_time = env.get_time_since_start()
    current_time = start_time
    
    print("Start simulation...")
    print(f"前 {pd_control_duration} 秒使用PD控制，之后切换到WBC控制")

    while current_time - start_time < scaled_duration:
        start_time_env = current_time
        start_time_wall = time.time()

        
        sim_base_pose = np.concatenate((
            sim_robot.get_base_position(), 
            sim_robot.get_base_quaternion()
        ))
        sim_base_velocity = np.concatenate((
            sim_robot.get_base_linear_velocity(),
            sim_robot.get_base_angular_velocity()
        ))
        sim_joint_positions = sim_robot.get_joint_positions()
        sim_joint_velocities = sim_robot.get_joint_velocities()
        
        
        robwrapper.set_state(
            sim_base_pose, sim_base_velocity, 
            sim_joint_positions, sim_joint_velocities
        )

        
        if use_pd_control and (current_time - start_time >= pd_control_duration):
            use_pd_control = False
            print(f"时间 {current_time:.2f}s: 切换到WBC控制模式")
        
        
        if use_pd_control:
            pd_progress = (current_time - start_time) / pd_control_duration
            if pd_progress % 0.5 < 0.01:  
                print(f"PD控制进度: {pd_progress*100:.1f}%")
                
            actions = pd_controller.compute_control(sim_joint_positions, sim_joint_velocities)
            
        else:
            try:
                actions = controller.compute_joint_actions(robwrapper, command)
                print(f"[WBC] 动作计算成功: {len(actions)} 个元素")
                
            except Exception as e:
                print(f"[WBC] 计算错误: {e}")
                actions = pd_controller.compute_control(sim_joint_positions, sim_joint_velocities)
                print("回退到PD控制")


        sim_robot.apply_joint_actions(actions)

        env.step(sleep=False)

        current_time = env.get_time_since_start()
        
        expected_duration = current_time - start_time_env
        actual_duration = time.time() - start_time_wall
        if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)

    print("Simulation completed.")

if __name__ == '__main__':
    main(sys.argv[1:])