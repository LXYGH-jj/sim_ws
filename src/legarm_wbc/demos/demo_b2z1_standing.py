"""
@file demo_b2z1_standing.py
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
MAX_TIME_SECS = 10  # maximum time to run the robot.


def main(argv):
    # Load configuration file
    if len(argv) == 1:
        cfg_file = argv[0]
    else:
        raise RuntimeError("Usage: python3 ./demo.py /<config file within root folder>")
    
    with open(rootdir + cfg_file, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    print("model_filename: ", cfgs['sim_robot_variables']['xml_filename'])


     # Set constant control.
    configs = load_yaml(rootdir + cfg_file)
    timestep = configs["timestep"]
    scale = configs["scale"]
    duration = configs["duration"]
    scaled_duration = duration * scale

    # ! Create a Mujoco simulation environment before any robots !
    env = MujocoEnvWithGround(dt=timestep)

    # Create a robot instance for Mujoco simulation.  
    sim_setting = SimRobotSetting()
    # Initialize the robot setting from configuration file.
    sim_setting.initialize(rootdir, cfg_file)
    sim_robot = SimRobotInterface(sim_setting)

    # Add the robot to the env to update the internal structure of the robot.
    env.add_robot(sim_robot)

    # Create a hierarchical inverse dynamics controller for control.
    ctrl_setting = ControllerSetting()
    ctrl_setting.initialize(rootdir, cfg_file)
    robwrapper = RobotWrapper(ctrl_setting)
    # robwrapper.validate_functions()
    controller = WholeBodyController(ctrl_setting)

    # Create a simple planner
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

    start_time = env.get_time_since_start()
    current_time = start_time
    while current_time - start_time < scaled_duration:
        start_time_env = current_time
        start_time_wall = time.time()

        # Update robot state in robot dynamics
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

        # Compute joint torque from controller
        actions = controller.compute_joint_actions(robwrapper, command)
        print(f"[DEBUG] Actions computed successfully: {len(actions)} elements")
        print(f"[DEBUG] First few action values: {actions[:10] if len(actions) > 10 else actions}")
        print(f"[DEBUG] Action values range: min={min(actions) if actions else 0}, max={max(actions) if actions else 0}")

        # Apply torque to robot
        sim_robot.apply_joint_actions(actions)

        # Step the simulation environment
        env.step(sleep=False)

        current_time = env.get_time_since_start()
        # Add sleep time
        expected_duration = current_time - start_time_env
        actual_duration = time.time() - start_time_wall
        if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)

    print("Simulation completed.")

if __name__ == '__main__':
    main(sys.argv[1:])




