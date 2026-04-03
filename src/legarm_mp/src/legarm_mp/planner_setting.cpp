/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   planner_setting.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Source file for PlannerSetting class
 *  @date   April 16, 2022
 **/

#include "legarm_mp/planner_setting.hpp"

#include <iostream>

#include <yamlutils/yaml_cpp_fwd.hpp>

namespace legarm_mp {

void PlannerSetting::initialize(const std::string& rootdir, 
                                const std::string& cfg_file, 
                                const std::string& planner_vars_yaml)
{
    try {
        YAML::Node planner_cfg = YAML::LoadFile((rootdir + cfg_file).c_str());
        YAML::Node planner_vars = planner_cfg[planner_vars_yaml.c_str()];

        YAML::readParameter(planner_vars, "timestep", timestep_);

        // robot property
        YAML::readParameter(planner_vars, "num_legs", num_legs_);
        YAML::readParameter(planner_vars, "num_arms", num_arms_);
        YAML::readParameter(planner_vars, "num_joints", num_joints_);
        YAML::readParameter(planner_vars, "body_mass", body_mass_);
        YAML::readParameter(planner_vars, "body_inertia", body_inertia_);
        YAML::readParameter(planner_vars, "hip_rel_pos", hip_rel_pos_);
        YAML::readParameter(planner_vars, "base_name", base_name_);
        YAML::readParameter(planner_vars, "leg_names", leg_names_);
        YAML::readParameter(planner_vars, "arm_names", arm_names_);
        YAML::readParameter(planner_vars, "leg_endeff_names", leg_endeff_names_);
        YAML::readParameter(planner_vars, "arm_endeff_names", arm_endeff_names_);

        YAML::readParameter(planner_vars, "clearance_ratio", clearance_ratio_);
        YAML::readParameter(planner_vars, "fric_coef", fric_coef_);
        YAML::readParameter(planner_vars, "foot_offset", foot_offset_);
        YAML::readParameter(planner_vars, "foot_delta_x_limit", foot_delta_x_limit_);
        YAML::readParameter(planner_vars, "foot_delta_y_limit", foot_delta_y_limit_);
        YAML::readParameter(planner_vars, "body_kp", body_kp_);
        YAML::readParameter(planner_vars, "body_kd", body_kd_);
        YAML::readParameter(planner_vars, "body_max_acc", body_max_acc_);
        YAML::readParameter(planner_vars, "raibert_kp", raibert_kp_);
        YAML::readParameter(planner_vars, "body_init_pos", body_init_pos_);
        YAML::readParameter(planner_vars, "body_init_vel", body_init_vel_);
        YAML::readParameter(planner_vars, "joint_init_pos", joint_init_pos_);
        YAML::readParameter(planner_vars, "joint_init_vel", joint_init_vel_);
        YAML::readParameter(planner_vars, "foot_init_rel_pos", foot_init_rel_pos_);
        YAML::readParameter(planner_vars, "gripper_init_rel_pos", gripper_init_rel_pos_);
    }
    catch (std::runtime_error& e) {
        std::cout << "[legarm_mp/PlannerSetting::initialize]: Error reading parameter [" 
                  << e.what() << "]" << std::endl;
    }
}

const std::string& PlannerSetting::get(PlannerStringParam param) const
{
    switch (param) {
        case PlannerStringParam_BaseName :
            return base_name_;
            break;
        default :
            throw std::runtime_error("[legarm_mp/PlannerSetting::get] PlannerStringParam invalid");
            break;
    }
}

const std::vector<std::string>& PlannerSetting::get(PlannerStringVectorParam param) const
{
    switch (param) {
        case PlannerStringVectorParam_LegNames :
            return leg_names_;
            break;
        case PlannerStringVectorParam_ArmNames :
            return arm_names_;
            break;
        case PlannerStringVectorParam_LegEndEffNames :
            return leg_endeff_names_;
            break;
        case PlannerStringVectorParam_ArmEndEffNames :
            return arm_endeff_names_;
            break;
        default :
            throw std::runtime_error("[legarm_mp/PlannerSetting::get] PlannerStringVectorParam invalid");
            break;
    }
}

int PlannerSetting::get(PlannerIntParam param) const
{
    switch (param) {
        case PlannerIntParam_NumLegs : 
            return num_legs_;
            break;
        case PlannerIntParam_NumArms : 
            return num_arms_;
            break;
        case PlannerIntParam_NumJoints :
            return num_joints_;
            break;
        default :
            throw std::runtime_error("[legarm_mp/PlannerSetting::get] PlannerIntParam invalid");
            break;
    }
}

double PlannerSetting::get(PlannerDoubleParam param) const
{
    switch (param) {
        case PlannerDoubleParam_TimeStep : 
            return timestep_;
            break;
        case PlannerDoubleParam_BodyMass :
            return body_mass_;
            break;
        case PlannerDoubleParam_ClearanceRatio :
            return clearance_ratio_;
            break;
        case PlannerDoubleParam_FricCoef : 
            return fric_coef_;
            break;
        case PlannerDoubleParam_FootOffset :
            return foot_offset_;
            break;
        case PlannerDoubleParam_FootDeltaxLimit :
            return foot_delta_x_limit_;
            break;
        case PlannerDoubleParam_FootDeltayLimit :
            return foot_delta_y_limit_;
            break;
        default :
            throw std::runtime_error("[legarm_mp/PlannerSetting::get] PlannerDoubleParam invalid");
            break;
    }
}

const PlannerSetting::VectorXd& PlannerSetting::get(PlannerVectorParam param) const
{
    switch (param) {
        case PlannerVectorParam_BodyKp : 
            return body_kp_;
            break;
        case PlannerVectorParam_BodyKd :
            return body_kd_;
            break;
        case PlannerVectorParam_BodyMaxAcc :
            return body_max_acc_;
            break;
        case PlannerVectorParam_RaibertKp :
            return raibert_kp_;
            break;
        case PlannerVectorParam_BodyInitPos :
            return body_init_pos_;
            break;
        case PlannerVectorParam_BodyInitVel :
            return body_init_vel_;
            break;
        case PlannerVectorParam_JointInitPos :
            return joint_init_pos_;
            break;
        case PlannerVectorParam_JointInitVel :
            return joint_init_vel_;
            break;
        default :
            throw std::runtime_error("[legarm_mp/PlannerSetting::get] PlannerVectorParam invalid");
            break;
    }
}

const PlannerSetting::MatrixXd& PlannerSetting::get(PlannerMatrixParam param) const
{
    switch (param) {
        case PlannerMatrixParam_BodyInertia :
            return body_inertia_;
            break;
        case PlannerMatrixParam_HipRelPos :
            return hip_rel_pos_;
            break;
        case PlannerMatrixParam_FootInitRelPos :
            return foot_init_rel_pos_;
            break;
        case PlannerMatrixParam_GripperInitRelPos :
            return gripper_init_rel_pos_;
            break;
        default :
            throw std::runtime_error("[legarm_mp/PlannerSetting::get] PlannerMatrixParam invalid");
            break;
    }
}

}  // end legarm_mp namespace