/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   motion_planner.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Header file for MotionPlanner class
 *  @date   August 04, 2022
 **/

#pragma once

#include <vector>
#include <string>
#include <map>

#include <Eigen/Dense>

#include <quad_gait/gait_scheduler.hpp>

#include "legarm_mp/planner_setting.hpp"
#include "legarm_mp/foot_swing_trajectory.hpp"
#include "legarm_mp/qp_force_optimizer.hpp"

namespace legarm_mp {

struct LinkStateRef
{
    Eigen::Vector3d position;
    Eigen::Matrix3d orientation;
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d force;
    Eigen::Vector3d torque;
};

class MotionPlanner
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::Matrix3d Matrix3d;
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::AngleAxisd AngleAxisd;
    typedef Eigen::Quaterniond Quaterniond;
    typedef Eigen::Ref<Vector3d>               RefVector3d;
    typedef Eigen::Ref<VectorXd>               RefVectorXd;
    typedef const Eigen::Ref<const Vector3d>&  ConstRefVector3d;
    typedef const Eigen::Ref<const VectorXd>&  ConstRefVectorXd;

    MotionPlanner(const PlannerSetting& setting);

    void setDesiredBodyLinearVelocity(ConstRefVector3d des_body_lin_vel);
    void setDesiredBodyAngularVelocity(ConstRefVector3d des_body_ang_vel);
    void setDesiredGripperLinearVelocity(ConstRefVector3d des_gripper_lin_vel);
    void setDesiredGripperAngularVelocity(ConstRefVector3d des_gripper_ang_vel);

    /**
     * @brief Compute task references according to current time and robot state.
     * 
     * @param base_pos Current base pose.
     * @param base_vel Current base linear and angular velocity.
     * @param gait_data Current gait data.
     */
    void computeTaskReferences(ConstRefVectorXd base_pos, 
                               ConstRefVectorXd base_vel, 
                               quad_gait::GaitData& gait_data);

    const Vector3d& getBodyPositionReference() const;
    const Matrix3d& getBodyOrientationReference() const;
    const Vector3d& getBodyEulerRPYReference() const;
    const Vector3d& getBodyLinearVelocityReference() const;
    const Vector3d& getBodyAngularVelocityReference() const;
    const Vector3d& getBodyEulerRPYRateReference() const;

    const Vector3d& getFootPositionReference(const std::string& foot_name) const;
    const Matrix3d& getFootOrientationReference(const std::string& foot_name) const;
    const Vector3d& getFootLinearVelocityReference(const std::string& foot_name) const;
    const Vector3d& getFootAngularVelocityReference(const std::string& foot_name) const;
    const Vector3d& getFootForceReference(const std::string& foot_name) const;
    const Vector3d& getFootTorqueReference(const std::string& foot_name) const;

    const Vector3d& getGripperPositionReference(const std::string& gripper_name) const;
    const Matrix3d& getGripperOrientationReference(const std::string& gripper_name) const;
    const Vector3d& getGripperEulerRPYReference(const std::string& gripper_name) const;
    const Vector3d& getGripperLinearVelocityReference(const std::string& gripper_name) const;
    const Vector3d& getGripperAngularVelocityReference(const std::string& gripper_name) const;
    const Vector3d& getGripperEulerRPYRateReference(const std::string& gripper_name) const;
    const Vector3d& getGripperForceReference(const std::string& gripper_name) const;
    const Vector3d& getGripperTorqueReference(const std::string& gripper_name) const;

private:
    double timestep_;

    // robot property
    int num_legs_;
    int num_arms_;
    int num_joints_;
    double body_mass_;
    Matrix3d body_inertia_;
    MatrixXd hip_rel_pos_;
    std::string base_name_;
    std::vector<std::string> leg_names_;
    std::vector<std::string> arm_names_;
    std::vector<std::string> leg_endeff_names_;
    std::vector<std::string> arm_endeff_names_;
    std::map<std::string, int> leg_endff_name_id_map_;
    std::map<std::string, int> arm_endff_name_id_map_;

    // user command
    Vector3d des_body_lin_vel_lcl_, des_body_ang_vel_lcl_;
    Vector3d des_gripper_lin_vel_lcl_, des_gripper_ang_vel_lcl_;

    // current body state
    Vector3d cur_body_lin_vel_lcl_, cur_body_ang_vel_lcl_;
    Vector3d cur_body_lin_vel_wrd_, cur_body_ang_vel_wrd_;
    Vector3d cur_body_pos_, cur_body_rpy_;
    Quaterniond cur_body_quat_;
    Matrix3d cur_body_orn_, cur_body_orn_z_;

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

    // task references
    Vector3d ref_body_pos_, ref_body_rpy_;
    Matrix3d ref_body_orn_;
    Quaterniond ref_body_quat_;
    Vector3d ref_body_lin_vel_wrd_, ref_body_ang_vel_wrd_;

    Vector3d ref_gripper_rpy_;

    // swing trajectory interpolation
    std::vector<FootSwingTrajectory> foot_trajs_;
    // store planned foot trajectory and force
    std::vector<LinkStateRef> foot_refs_;

    // store planned gripper trajectory and force
    std::vector<LinkStateRef> gripper_refs_;

    VectorXd cur_q_, cur_dq_, des_q_, des_dq_, des_ddq_;
    MatrixXd rel_feet_pos_;
    QPForceOptimizer optimizer_;
};

}  // end legarm_mp namespace 