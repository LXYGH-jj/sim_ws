/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   planner_setting.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Header file for PlannerSetting class
 *  @date   April 16, 2022
 **/

#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "legarm_mp/planner_params.hpp"

namespace legarm_mp {

class PlannerSetting
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::MatrixXd MatrixXd;

    PlannerSetting() {}
    ~PlannerSetting() {}

    void initialize(const std::string& rootdir, 
                    const std::string& cfg_file, 
                    const std::string& planner_vars_yaml="planner_variables");

    const std::string& get(PlannerStringParam param) const;
    const std::vector<std::string>& get(PlannerStringVectorParam param) const;
    int get(PlannerIntParam param) const;
    double get(PlannerDoubleParam param) const;
    const VectorXd& get(PlannerVectorParam param) const;
    const MatrixXd& get(PlannerMatrixParam param) const;

private:
    double timestep_;

    // robot property
    int num_legs_;
    int num_arms_;
    int num_joints_;
    double body_mass_;
    MatrixXd body_inertia_;
    MatrixXd hip_rel_pos_;
    std::string base_name_;
    std::vector<std::string> leg_names_;
    std::vector<std::string> arm_names_;
    std::vector<std::string> leg_endeff_names_;
    std::vector<std::string> arm_endeff_names_;

    double clearance_ratio_;
    double fric_coef_;
    double foot_offset_;
    double foot_delta_x_limit_;
    double foot_delta_y_limit_;
    VectorXd body_kp_;
    VectorXd body_kd_;
    VectorXd body_max_acc_;
    VectorXd raibert_kp_;
    VectorXd body_init_pos_;
    VectorXd body_init_vel_;
    VectorXd joint_init_pos_;
    VectorXd joint_init_vel_;
    MatrixXd foot_init_rel_pos_;
    MatrixXd gripper_init_rel_pos_;
};

}  // end legarm_mp namespace