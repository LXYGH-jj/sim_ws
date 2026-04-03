"""
@file demo_aliengovx300_head_stabilization.py
@package legarm_mp
@author Jun LI (junlileeds@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
@date 2022-09-05
"""
###########################
import yaml

import os
import sys
import inspect
import time
import numpy as np

import csv
from scipy.spatial.transform import Rotation as R

from commutils.yaml_parser import load_yaml

# from limbsim.env import BulletEnvWithGround
# from limbsim.sim_robot_setting import SimRobotSetting
# from limbsim.sim_robot_interface import SimRobotInterface
# from limbsim.joint_controller import JointController

from mjcsim.env import MujocoEnv
from mjcsim.sim_robot_setting import SimRobotSetting
from mjcsim.sim_robot_interface import SimRobotInterface
from mjcsim.joint_controller import JointController

from legarm_wbc.controller_setting import ControllerSetting
from legarm_wbc.robot_wrapper import RobotWrapper
from legarm_wbc.whole_body_controller import WholeBodyController
from legarm_wbc.command import HighLevelCommand

from quad_gait_pywrap import UserCommand, GaitScheduler, GaitType
from legarm_mp_pywrap import PlannerSetting, MotionPlanner
from traj_gen.pose_planner import PosePlanner
# from traj_gen.gripping_planner import GrippingPlanner

# ============================================================================
class PDController:
    """A simple PD controller for initial stabilization"""
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
        
        desired_velocities = np.zeros_like(current_velocities)
        position_errors = self.desired_positions - current_positions
        velocity_errors = desired_velocities - current_velocities
        torques = self.joint_kp * position_errors + self.joint_kd * velocity_errors
        
        actions = []
        for idx, jn in enumerate(self.joint_names):
            actions.extend([
                self.desired_positions[idx],  
                self.joint_kp[idx],           
                0.0,                         
                self.joint_kd[idx],           
                torques[idx]                  
            ])
        return actions

# absolute directory of this package
rootdir = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))))


def main(argv):
    # Load configuration file
    if len(argv) == 1:
        cfg_file = argv[0]
    else:
        raise RuntimeError("Usage: python3 ./demo.py /<config file within root folder>")
    
    with open(rootdir + cfg_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    print("model_filename: ", configs['sim_robot_variables']['xml_filename'])


    # configs = load_yaml(rootdir + cfg_file)

    timestep = configs["timestep"]
    scale = configs["scale"]
    duration = configs["duration"]
    scaled_duration = duration * scale

    # ! Create a PyBullet simulation environment before any robots !
    # env = BulletEnvWithGround(dt=timestep)
    env = MujocoEnv(dt=timestep)

    # env.configure_gui(
    #     camera_distance=0.85, 
    #     camera_yaw=45, 
    #     camera_pitch=-15, 
    #     camera_target_position=[0.10, 0.00, 0.50]
    # )

    # Create a robot instance for PyBullet.
    sim_setting = SimRobotSetting()
    sim_setting.initialize(rootdir, cfg_file)
    sim_robot = SimRobotInterface(sim_setting)
    # joint_controller = JointController(sim_robot.joint_names[:12], 
    #                                    sim_robot.joint_torque_limits[:12])

    # Add the robot to the env to update the internal structure of the robot.
    env.add_robot(sim_robot)

    # Create a hierarchical inverse dynamics controller for control.
    ctrl_setting = ControllerSetting()
    ctrl_setting.initialize(rootdir, cfg_file)
    robwrapper = RobotWrapper(ctrl_setting)
    controller = WholeBodyController(ctrl_setting)

    # Create high level command
    command = HighLevelCommand()
    des_task_hierarchy = configs["planner_variables"]["task_hierarchy"]
    des_base_pose = {}
    des_joint_posture = {}
    des_joint_posture["joint"] = {
        "pos": np.array(configs["planner_variables"]["joint_init_pos"]),
        "lin_vel": np.array(configs["planner_variables"]["joint_init_pos"]),
        "lin_acc": np.zeros(len(np.array(configs["planner_variables"]["joint_init_pos"]))),
    }
    des_stance_legs = [] 
    des_swing_legs = []
    des_stance_arms = []
    des_swing_arms = []
    des_stance_leg_endeff_wrenches = {}
    des_swing_leg_endeff_poses = {}
    des_stance_arm_endeff_wrenches = {}
    des_swing_arm_endeff_poses = {}
    command.set_desired_task_hierarchy(des_task_hierarchy)
    command.set_desired_joint_posture(des_joint_posture)



    # Create a SRBM based motion planner
    pl_setting = PlannerSetting()
    pl_setting.initialize(rootdir, cfg_file)
    planner = MotionPlanner(pl_setting)

    # Create a gait scheduler
    user_cmd = UserCommand()
    user_cmd.gait_type = GaitType.STAND
    user_cmd.gait_override = 1
    scheduler = GaitScheduler(timestep, user_cmd)

    # ============================================================================
    # Create PD controller for initial stabilization
    pd_controller = PDController(
        joint_kp=np.array(configs["control_variables"]["joint_kp"]),
        joint_kd=np.array(configs["control_variables"]["joint_kd"]),
        joint_names=configs["control_variables"]["joint_names"]
    )
    # 使用 control_variables 中的期望关节位置作为 PD 控制目标
    pd_controller.set_desired_positions(np.array(configs["control_variables"]["joint_des_pos"]))
    
    # PD control parameters
    pd_control_duration = 3.0  # PD 控制持续时间 [s]
    use_pd_control = True      # 是否使用 PD 控制
    pd_print_counter = 0       # PD 进度打印计数器

    # Create offline trajectory planners for base and gripper
    body_traj_planner = PosePlanner(dt=timestep)
    gripper_traj_planner = PosePlanner(dt=timestep)

    #########################################################################################
    # finger_traj_planner = GrippingPlanner(dt=timestep, 
    #     close_pos=[0.021, -0.021], open_pos=[0.057, -0.057])

    # pose 0
    body_pos_0 = np.array([0., 0., 0.40])
    body_orn_0 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    gripper_pos_0 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_0 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_0 = 0  # 0 close, 1 open

    # pose 1
    body_pos_1 = np.array([-0.3, 0., 0.40])
    body_orn_1 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    gripper_pos_1 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_1 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_1 = 0  # 0 close, 1 open

    # pose 2
    body_pos_2 = np.array([-0.3, 0., 0.40])
    body_orn_2 = np.array([
        [1, 0, 0],
        [0, np.cos(np.pi/6), -np.sin(np.pi/6)],
        [0, np.sin(np.pi/6), np.cos(np.pi/6)]
    ])
    gripper_pos_2 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_2 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_2 = 0  # 0 close, 1 open
    # pose 3
    body_pos_3 = np.array([-0.3, 0., 0.40])
    body_orn_3 = np.array([
        [1, 0, 0],
        [0, np.cos(-np.pi/6), -np.sin(-np.pi/6)],
        [0, np.sin(-np.pi/6), np.cos(-np.pi/6)]
    ])
    gripper_pos_3 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_3 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_3 = 0  # 0 close, 1 open

    # pose 4
    body_pos_4 = np.array([-0.3, 0.2, 0.40])
    body_orn_4 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    gripper_pos_4 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_4 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_4 = 0  # 0 close, 1 open
    # pose 5
    body_pos_5 = np.array([-0.3, -0.2, 0.40])
    body_orn_5 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    gripper_pos_5 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_5 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_5 = 0  # 0 close, 1 open

    # pose 6
    body_pos_6 = np.array([-0.3, 0., 0.40])
    body_orn_6 = np.array([
        [np.cos(np.pi/6), 0, np.sin(np.pi/6)],
        [0, 1, 0],
        [-np.sin(np.pi/6), 0, np.cos(np.pi/6)]
    ])
    gripper_pos_6 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_6 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_6 = 0  # 0 close, 1 open
    # pose 7
    body_pos_7 = np.array([-0.3, 0., 0.40])
    body_orn_7 = np.array([
        [np.cos(-np.pi/6), 0, np.sin(-np.pi/6)],
        [0, 1, 0],
        [-np.sin(-np.pi/6), 0, np.cos(-np.pi/6)]
    ])
    gripper_pos_7 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_7 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_7 = 0  # 0 close, 1 open

    # pose 8
    body_pos_8 = np.array([-0.3, 0., 0.30])
    body_orn_8 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    gripper_pos_8 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_8 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_8 = 0  # 0 close, 1 open
    # pose 9
    body_pos_9 = np.array([-0.3, 0., 0.50])
    body_orn_9 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    gripper_pos_9 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_9 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_9 = 0  # 0 close, 1 open

    # pose 10
    body_pos_10 = np.array([-0.3, 0., 0.40])
    body_orn_10 = np.array([
        [np.cos(np.pi/6), -np.sin(np.pi/6), 0],
        [np.sin(np.pi/6), np.cos(np.pi/6), 0],
        [0, 0, 1]
    ])
    gripper_pos_10 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_10 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_10 = 0  # 0 close, 1 open

    # pose 11
    body_pos_11 = np.array([-0.3, 0., 0.40])
    body_orn_11 = np.array([
        [np.cos(-np.pi/6), -np.sin(-np.pi/6), 0],
        [np.sin(-np.pi/6), np.cos(-np.pi/6), 0],
        [0, 0, 1]
    ])
    gripper_pos_11 = np.array([0.5627, 0.0020, 0.4304])
    gripper_orn_11 = np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    # finger_state_11 = 0  # 0 close, 1 open


    body_traj_planner.plan_cartesian_motion(
        body_pos_0, body_orn_0, body_pos_0, body_orn_0, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_0, body_orn_0, body_pos_1, body_orn_1, 2.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_1, body_orn_1, body_pos_2, body_orn_2, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_2, body_orn_2, body_pos_3, body_orn_3, 2.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_3, body_orn_3, body_pos_1, body_orn_1, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_1, body_orn_1, body_pos_4, body_orn_4, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_4, body_orn_4, body_pos_5, body_orn_5, 2.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_5, body_orn_5, body_pos_1, body_orn_1, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_1, body_orn_1, body_pos_6, body_orn_6, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_6, body_orn_6, body_pos_7, body_orn_7, 2.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_7, body_orn_7, body_pos_1, body_orn_1, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_1, body_orn_1, body_pos_8, body_orn_8, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_8, body_orn_8, body_pos_9, body_orn_9, 2.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_9, body_orn_9, body_pos_1, body_orn_1, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_1, body_orn_1, body_pos_10, body_orn_10, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_10, body_orn_10, body_pos_11, body_orn_11, 2.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_11, body_orn_11, body_pos_1, body_orn_1, 1.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_1, body_orn_1, body_pos_0, body_orn_0, 2.*scale)
    body_traj_planner.plan_cartesian_motion(
        body_pos_0, body_orn_0, body_pos_0, body_orn_0, 1.*scale)
    body_traj_planner.calculate_motion_velocity()

    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_0, gripper_orn_0, gripper_pos_0, gripper_orn_0, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_0, gripper_orn_0, gripper_pos_1, gripper_orn_1, 2.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_1, gripper_orn_1, gripper_pos_2, gripper_orn_2, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_2, gripper_orn_2, gripper_pos_3, gripper_orn_3, 2.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_3, gripper_orn_3, gripper_pos_1, gripper_orn_1, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_1, gripper_orn_1, gripper_pos_4, gripper_orn_4, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_4, gripper_orn_4, gripper_pos_5, gripper_orn_5, 2.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_5, gripper_orn_5, gripper_pos_1, gripper_orn_1, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_1, gripper_orn_1, gripper_pos_6, gripper_orn_6, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_6, gripper_orn_6, gripper_pos_7, gripper_orn_7, 2.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_7, gripper_orn_7, gripper_pos_1, gripper_orn_1, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_1, gripper_orn_1, gripper_pos_8, gripper_orn_8, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_8, gripper_orn_8, gripper_pos_9, gripper_orn_9, 2.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_9, gripper_orn_9, gripper_pos_1, gripper_orn_1, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_1, gripper_orn_1, gripper_pos_10, gripper_orn_10, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_10, gripper_orn_10, gripper_pos_11, gripper_orn_11, 2.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_11, gripper_orn_11, gripper_pos_1, gripper_orn_1, 1.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_1, gripper_orn_1, gripper_pos_0, gripper_orn_0, 2.*scale)
    gripper_traj_planner.plan_cartesian_motion(
        gripper_pos_0, gripper_orn_0, gripper_pos_0, gripper_orn_0, 1.*scale)
    gripper_traj_planner.calculate_motion_velocity()

    # finger_traj_planner.plan_gripping_motion(finger_state_0, finger_state_0, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_0, finger_state_1, 2.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_1, finger_state_2, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_2, finger_state_3, 2.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_3, finger_state_1, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_1, finger_state_4, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_4, finger_state_5, 2.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_5, finger_state_1, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_1, finger_state_6, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_6, finger_state_7, 2.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_7, finger_state_1, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_1, finger_state_8, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_8, finger_state_9, 2.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_9, finger_state_1, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_1, finger_state_10, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_10, finger_state_11, 2.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_11, finger_state_1, 1.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_1, finger_state_0, 2.*scale)
    # finger_traj_planner.plan_gripping_motion(finger_state_0, finger_state_0, 1.*scale)


    csv_file = csv.writer(open(rootdir + "/data/b2z1_head_stabilizing.csv", "w"))
    data_headings = [
        "time", 
        # from scheduler
        "cont_state_fl", "cont_state_fr", "cont_state_rl", "cont_state_rr", 
        # from planner
        "ref_body_pos_x", "ref_body_pos_y", "ref_body_pos_z", 
        "ref_body_lin_vel_x", "ref_body_lin_vel_y", "ref_body_lin_vel_z",
        "ref_body_euler_r", "ref_body_euler_p", "ref_body_euler_y", 
        "ref_body_euler_dr", "ref_body_euler_dp", "ref_body_euler_dy", 
        "ref_foot_fl_pos_x", "ref_foot_fl_pos_y", "ref_foot_fl_pos_z", 
        "ref_foot_fr_pos_x", "ref_foot_fr_pos_y", "ref_foot_fr_pos_z", 
        "ref_foot_rl_pos_x", "ref_foot_rl_pos_y", "ref_foot_rl_pos_z", 
        "ref_foot_rr_pos_x", "ref_foot_rr_pos_y", "ref_foot_rr_pos_z", 
        "ref_foot_fl_lin_vel_x", "ref_foot_fl_lin_vel_y", "ref_foot_fl_lin_vel_z", 
        "ref_foot_fr_lin_vel_x", "ref_foot_fr_lin_vel_y", "ref_foot_fr_lin_vel_z", 
        "ref_foot_rl_lin_vel_x", "ref_foot_rl_lin_vel_y", "ref_foot_rl_lin_vel_z", 
        "ref_foot_rr_lin_vel_x", "ref_foot_rr_lin_vel_y", "ref_foot_rr_lin_vel_z", 
        "ref_foot_fl_frc_x", "ref_foot_fl_frc_y", "ref_foot_fl_frc_z",
        "ref_foot_fr_frc_x", "ref_foot_fr_frc_y", "ref_foot_fr_frc_z",
        "ref_foot_rl_frc_x", "ref_foot_rl_frc_y", "ref_foot_rl_frc_z",
        "ref_foot_rr_frc_x", "ref_foot_rr_frc_y", "ref_foot_rr_frc_z",
        "ref_gripper_pos_x", "ref_gripper_pos_y", "ref_gripper_pos_z", 
        "ref_gripper_lin_vel_x", "ref_gripper_lin_vel_y", "ref_gripper_lin_vel_z", 
        "ref_gripper_euler_r", "ref_gripper_euler_p", "ref_gripper_euler_y",
        "ref_gripper_euler_dr", "ref_gripper_euler_dp", "ref_gripper_euler_dy",
        "ref_gripper_frc_x", "ref_gripper_frc_y", "ref_gripper_frc_z", 
        # from controller
        "wbc_jnt_trq_fl_hip", "wbc_jnt_trq_fl_upper", "wbc_jnt_trq_fl_lower", 
        "wbc_jnt_trq_fr_hip", "wbc_jnt_trq_fr_upper", "wbc_jnt_trq_fr_lower", 
        "wbc_jnt_trq_rl_hip", "wbc_jnt_trq_rl_upper", "wbc_jnt_trq_rl_lower", 
        "wbc_jnt_trq_rr_hip", "wbc_jnt_trq_rr_upper", "wbc_jnt_trq_rr_lower", 
############################################################################################
        "wbc_jnt_trq_waist", "wbc_jnt_trq_shoulder", "wbc_jnt_trq_elbow", 
        "wbc_jnt_trq_wrist_angle", "wbc_jnt_trq_wrist_rotate","wbc_jnt_trq_wrist2",
###############################################################################################
        "wbc_foot_frc_fl_x", "wbc_foot_frc_fl_y", "wbc_foot_frc_fl_z",
        "wbc_foot_frc_fr_x", "wbc_foot_frc_fr_y", "wbc_foot_frc_fr_z",
        "wbc_foot_frc_rl_x", "wbc_foot_frc_rl_y", "wbc_foot_frc_rl_z",
        "wbc_foot_frc_rr_x", "wbc_foot_frc_rr_y", "wbc_foot_frc_rr_z",
        # from simulator
        "sim_base_pos_x", "sim_base_pos_y", "sim_base_pos_z",
        "sim_base_lin_vel_x", "sim_base_lin_vel_y", "sim_base_lin_vel_z",
        "sim_base_euler_r", "sim_base_euler_p", "sim_base_euler_y", 
        "sim_base_euler_dr", "sim_base_euler_dp", "sim_base_euler_dy",
        "sim_foot_fl_pos_x", "sim_foot_fl_pos_y", "sim_foot_fl_pos_z", 
        "sim_foot_fr_pos_x", "sim_foot_fr_pos_y", "sim_foot_fr_pos_z", 
        "sim_foot_rl_pos_x", "sim_foot_rl_pos_y", "sim_foot_rl_pos_z", 
        "sim_foot_rr_pos_x", "sim_foot_rr_pos_y", "sim_foot_rr_pos_z", 
        "sim_foot_fl_lin_vel_x", "sim_foot_fl_lin_vel_y", "sim_foot_fl_lin_vel_z", 
        "sim_foot_fr_lin_vel_x", "sim_foot_fr_lin_vel_y", "sim_foot_fr_lin_vel_z", 
        "sim_foot_rl_lin_vel_x", "sim_foot_rl_lin_vel_y", "sim_foot_rl_lin_vel_z", 
        "sim_foot_rr_lin_vel_x", "sim_foot_rr_lin_vel_y", "sim_foot_rr_lin_vel_z",
        "sim_foot_fl_frc_z", "sim_foot_fr_frc_z","sim_foot_rl_frc_z","sim_foot_rr_frc_z",
        "sim_gripper_pos_x", "sim_gripper_pos_y", "sim_gripper_pos_z", 
        "sim_gripper_lin_vel_x", "sim_gripper_lin_vel_y", "sim_gripper_lin_vel_z", 
        "sim_gripper_euler_r", "sim_gripper_euler_p", "sim_gripper_euler_y",
        "sim_gripper_euler_dr", "sim_gripper_euler_dp", "sim_gripper_euler_dy",
    ]
    csv_file.writerow(data_headings)
    data = []
    counter = 0


    start_time = env.get_time_since_start()
    current_time = start_time
    #env.start_video_recording(rootdir + "/video/b2z1_head_stabilizing.mp4")
    while current_time - start_time < scaled_duration:
        # ========================================================================
        # 添加启动信息显示（在 while 循环内第一行添加）
        if current_time == start_time:
            print("Start simulation...")
            print(f"前 {pd_control_duration} 秒使用 PD 控制，之后切换到 WBC 控制")
        # ========================================================================
        
        start_time_env = current_time
        start_time_wall = time.time()

        # Update gait
        if current_time > 1.*scale:
            user_cmd.gait_type = GaitType.TROT_WALK
        if current_time > 3.*scale:
            user_cmd.gait_type = GaitType.STAND
        if current_time > 23.*scale:
            user_cmd.gait_type = GaitType.TROT_WALK
        if current_time > 25.*scale:
            user_cmd.gait_type = GaitType.STAND
        scheduler.step()

        # ========================================================================
        # Check if should switch from PD to WBC
        if use_pd_control and (current_time - start_time >= pd_control_duration):
            use_pd_control = False

            # 🔍 添加：打印 PD 稳定后的末端位置
            gripper_pose = sim_robot.get_link_pose("link06")
            base_pos = sim_robot.get_base_position()
            
            # 计算相对于基座的末端位置
            gripper_pos_world = gripper_pose[0:3]
            gripper_rel_pos = gripper_pos_world - base_pos
            
            print(f"时间 {current_time:.2f}s: 切换到 WBC 控制模式")
            print(f"  PD 稳定后机械臂末端位置 (世界坐标系): [{gripper_pos_world[0]:.4f}, {gripper_pos_world[1]:.4f}, {gripper_pos_world[2]:.4f}]")
            print(f"  PD 稳定后机械臂末端位置 (基座坐标系): [{gripper_rel_pos[0]:.4f}, {gripper_rel_pos[1]:.4f}, {gripper_rel_pos[2]:.4f}]")
            print(f"  请将此值复制到 YAML 的 gripper_init_rel_pos")
        # ==========================================================
            print(f"时间 {current_time:.2f}s: 切换到 WBC 控制模式")
        
        # # PD control progress display
        # if use_pd_control:
        #     pd_progress = (current_time - start_time) / pd_control_duration
        #     if int(pd_progress * 100) % 10 == 0:  # 每 10% 显示一次
        #         print(f"PD 控制进度：{pd_progress*100:.1f}%")
        # 在 PD 控制器初始化时添加
        

        # 在主循环中修改为
        if use_pd_control:
            pd_print_counter += 1
            if pd_print_counter % 250 == 0:  # 每 0.5 秒显示一次（500Hz * 0.5s）
                pd_progress = (current_time - start_time) / pd_control_duration
                print(f"PD 控制进度：{pd_progress*100:.1f}%")
        # ========================================================================


        # Update desired velocity
        step = int(current_time/timestep)
        des_body_lin_vel = body_traj_planner.traj_lin_vel_lcl[step]
        des_body_ang_vel = body_traj_planner.traj_ang_vel_lcl[step]
        des_gripper_lin_vel = gripper_traj_planner.traj_lin_vel_lcl[step]
        des_gripper_ang_vel = gripper_traj_planner.traj_ang_vel_lcl[step]
        #des_finger_pos = finger_traj_planner.traj_pos[step]

        planner.setDesiredBodyLinearVelocity(des_body_lin_vel)
        planner.setDesiredBodyAngularVelocity(des_body_ang_vel)
        planner.setDesiredGripperLinearVelocity(des_gripper_lin_vel)
        planner.setDesiredGripperAngularVelocity(des_gripper_ang_vel)
        #sim_robot.apply_joint_positions(["left_finger", "right_finger"], des_finger_pos)

        # Collect robot state for robot dynamics
        base_pose = np.concatenate((
            sim_robot.get_base_position(),
            sim_robot.get_base_quaternion()
        ))
        base_velocity = np.concatenate((
            sim_robot.get_base_linear_velocity(),
            sim_robot.get_base_angular_velocity()
        ))
        joint_positions = sim_robot.get_joint_positions()
        joint_velocities = sim_robot.get_joint_velocities()
        contact_states = sim_robot.get_limb_contact_states()

        # Compute command from planner
        planner.computeTaskReferences(
            base_pose, base_velocity, scheduler.gaitData()
        )

        # Update robot dynamics
        robwrapper.set_state(
            base_pose, base_velocity, 
            joint_positions, joint_velocities
        )

        # Update high level command
        des_base_pose[configs["planner_variables"]["base_name"]] = {
            "pos": planner.getBodyPositionReference(), 
            "orn": planner.getBodyOrientationReference(), 
            "lin_vel": planner.getBodyLinearVelocityReference(), 
            "ang_vel": planner.getBodyAngularVelocityReference(), 
            "lin_acc": np.zeros(3), 
            "ang_acc": np.zeros(3)
        }
        command.set_desired_base_pose(des_base_pose)
        des_stance_legs.clear()
        des_swing_legs.clear()
        des_stance_arms.clear()
        des_swing_arms.clear()
        des_stance_leg_endeff_wrenches.clear()
        des_swing_leg_endeff_poses.clear()
        des_stance_arm_endeff_wrenches.clear()
        des_swing_arm_endeff_poses.clear()

        ######################################################################
        for i, state in enumerate(scheduler.gaitData().contact_state_scheduled):
            leg_name = configs["planner_variables"]["leg_names"][i]
            leg_endeff_name = configs["planner_variables"]["leg_endeff_names"][i]
            if state == 0:
                des_swing_legs.append(leg_name)
                des_swing_leg_endeff_poses[leg_endeff_name] = {
                        "pos": planner.getFootPositionReference(leg_endeff_name), 
                        "lin_vel": planner.getFootLinearVelocityReference(leg_endeff_name), 
                        "lin_acc": np.zeros(3), 
                    }
            elif state == 1:
                des_stance_legs.append(leg_name)
                des_stance_leg_endeff_wrenches[leg_endeff_name] = {
                    "force": planner.getFootForceReference(leg_endeff_name),
                    "torque": np.zeros(3),
                }
            else:
                raise RuntimeError("Unknown leg state!")
        command.set_desired_stance_legs(des_stance_legs)
        command.set_desired_swing_legs(des_swing_legs)
        command.set_desired_stance_leg_endeff_wrenches(des_stance_leg_endeff_wrenches)
        command.set_desired_swing_leg_endeff_poses(des_swing_leg_endeff_poses)


        for i, state in enumerate(np.array([0])):
            arm_name = configs["planner_variables"]["arm_names"][i]
            arm_endeff_name = configs["planner_variables"]["arm_endeff_names"][i]
            if state == 0:
                des_swing_arms.append(arm_name)
                des_swing_arm_endeff_poses[arm_endeff_name] = {
                        "pos": planner.getGripperPositionReference(arm_endeff_name), 
                        "orn": planner.getGripperOrientationReference(arm_endeff_name),
                        "lin_vel": planner.getGripperLinearVelocityReference(arm_endeff_name), 
                        "ang_vel": planner.getGripperAngularVelocityReference(arm_endeff_name),
                        "lin_acc": np.zeros(3), 
                        "ang_acc": np.zeros(3),
                }
            elif state == 1:
                des_stance_arms.append(arm_name)
                des_stance_arm_endeff_wrenches[arm_endeff_name] = {
                    "force": planner.getGripperForceReference(arm_endeff_name),
                    "torque": np.zeros(3),
                }
            else:
                raise RuntimeError("Unknown arm state!")
        command.set_desired_stance_arms(des_stance_arms)
        command.set_desired_swing_arms(des_swing_arms)
        command.set_desired_stance_arm_endeff_wrenches(des_stance_leg_endeff_wrenches)
        command.set_desired_swing_arm_endeff_poses(des_swing_arm_endeff_poses)

        # Compute joint torque from controller
        # actions = controller.compute_joint_actions(robwrapper, command)
        # ========================================================================
        # Compute joint torque from controller
        if use_pd_control:
            # Use PD control for stabilization
            actions = pd_controller.compute_control(
                joint_positions, 
                joint_velocities
            )
        else:
            # Use WBC for motion control
            actions = controller.compute_joint_actions(robwrapper, command)
        # ========================================================================


        # # Apply torque to quadrupedal robot
        # leg_joint_pos = sim_robot.get_joint_positions()[:12]
        # leg_joint_vel = sim_robot.get_joint_velocities()[:12]
        # leg_joint_trq = joint_controller.convert_to_torque(
        #     leg_joint_pos, leg_joint_vel, actions[:5*12])
        # sim_robot.apply_joint_torques(
        #     configs["controller_variables"]["joint_names"][:12], 
        #     leg_joint_trq)

        # # Set arm's desired position
        # sim_robot.apply_joint_positions(
        #     configs["controller_variables"]["joint_names"][12:], 
        #     controller.des_q[7+12:])

        # Apply torque to robot
        sim_robot.apply_joint_actions(actions)

        # Step the simulation environment
        env.step(sleep=False)

        counter += 1
        if counter == 10:
            data.append(current_time)
            # from scheduler
            data.extend(scheduler.gaitData().contact_state_scheduled)
            # from planner
            data.extend(planner.getBodyPositionReference())
            data.extend(planner.getBodyLinearVelocityReference())
            data.extend(planner.getBodyEulerRPYReference())
            data.extend(planner.getBodyEulerRPYRateReference())
            data.extend(planner.getFootPositionReference("FL_foot"))
            data.extend(planner.getFootPositionReference("FR_foot"))
            data.extend(planner.getFootPositionReference("RL_foot"))
            data.extend(planner.getFootPositionReference("RR_foot"))
            data.extend(planner.getFootLinearVelocityReference("FL_foot"))
            data.extend(planner.getFootLinearVelocityReference("FR_foot"))
            data.extend(planner.getFootLinearVelocityReference("RL_foot"))
            data.extend(planner.getFootLinearVelocityReference("RR_foot"))
            data.extend(planner.getFootForceReference("FL_foot"))
            data.extend(planner.getFootForceReference("FR_foot"))
            data.extend(planner.getFootForceReference("RL_foot"))
            data.extend(planner.getFootForceReference("RR_foot"))
            data.extend(planner.getGripperPositionReference("link06"))
            data.extend(planner.getGripperLinearVelocityReference("link06"))
            data.extend(planner.getGripperEulerRPYReference("link06"))
            data.extend(planner.getGripperEulerRPYRateReference("link06"))
            data.extend(planner.getGripperForceReference("link06"))
            # from controller
            data.extend(controller.des_tau[6:])
            data.extend(controller.des_leg_endeff_forces["FL_foot"])
            data.extend(controller.des_leg_endeff_forces["FR_foot"])
            data.extend(controller.des_leg_endeff_forces["RL_foot"])
            data.extend(controller.des_leg_endeff_forces["RR_foot"])
            # from simulator
            data.extend(sim_robot.get_base_position())
            data.extend(sim_robot.get_base_linear_velocity())
            data.extend(sim_robot.get_base_euler_rpy())
            data.extend(sim_robot.get_base_angular_velocity())
            data.extend(sim_robot.get_link_pose("FL_foot")[0:3])
            data.extend(sim_robot.get_link_pose("FR_foot")[0:3])
            data.extend(sim_robot.get_link_pose("RL_foot")[0:3])
            data.extend(sim_robot.get_link_pose("RR_foot")[0:3])
            data.extend(sim_robot.get_link_velocity("FL_foot")[0:3])
            data.extend(sim_robot.get_link_velocity("FR_foot")[0:3])
            data.extend(sim_robot.get_link_velocity("RL_foot")[0:3])
            data.extend(sim_robot.get_link_velocity("RR_foot")[0:3])
            data.extend(sim_robot.get_limb_contact_forces()[0:4])
            data.extend(sim_robot.get_link_pose("link06")[0:3])
            data.extend(sim_robot.get_link_velocity("link06")[0:3])
            data.extend(R.from_quat(sim_robot.get_link_pose("link06")[3:7]).as_euler('xyz'))
            data.extend(R.from_quat(sim_robot.get_link_pose("link06")[3:7]).as_matrix().T.dot(sim_robot.get_link_velocity("link06")[3:6]))

            csv_file.writerow(data)
            data.clear()
            counter = 0

        current_time = env.get_time_since_start()
        # Add sleep time
        expected_duration = current_time - start_time_env
        actual_duration = time.time() - start_time_wall
        if actual_duration < expected_duration:
            time.sleep(expected_duration - actual_duration)
    # input("please enter to continue....")
    # env.stop_video_recording()

if __name__ == "__main__":
    main(sys.argv[1:])
