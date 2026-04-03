/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   motion_planner.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Source file for MotionPlanner class
 *  @date   August 04, 2022
 **/

#include "legarm_mp/motion_planner.hpp"

#include <cmath>

namespace legarm_mp {

namespace {

const double TOLERANCE = 1e-10;

typedef Eigen::Vector3d Vector3d;
typedef Eigen::Matrix3d Matrix3d;

Matrix3d skew(const Vector3d& v) {
    // Convert vector to skew-symmetric matrix
    Matrix3d M = Matrix3d::Zero();
    M << 0, -v[2], v[1],
         v[2], 0, -v[0], 
        -v[1], v[0], 0;
    return M;
}

Matrix3d ExpSO3(const Vector3d& w) {
    // Computes the vectorized exponential map for SO(3)
    Matrix3d A = skew(w);
    double theta = w.norm();
    if (theta < TOLERANCE) {
        return Matrix3d::Identity();
    } 
    Matrix3d R = Matrix3d::Identity() + (std::sin(theta)/theta)*A + 
        ((1-std::cos(theta))/(theta*theta))*A*A;
    return R;
}

Vector3d matrixToRpy(const Matrix3d& R) {
    // assert(R.isUnitary() && "R is not a unitary matrix");
    Vector3d rpy = R.eulerAngles(2,1,0).reverse();

    if (rpy[1] < - M_PI/2)
        rpy[1] += 2 * M_PI;
    
    if (rpy[1] > M_PI/2) {
        rpy[1] = M_PI - rpy[1];
        if (rpy[0] < 0.)
            rpy[0] += M_PI;
        else
            rpy[0] -= M_PI;
        rpy[2] -= M_PI;
    }

    return rpy;
}

}  // end namespace


MotionPlanner::MotionPlanner(const PlannerSetting& setting)
    : timestep_(setting.get(PlannerDoubleParam_TimeStep)), 
      num_legs_(setting.get(PlannerIntParam_NumLegs)),
      num_arms_(setting.get(PlannerIntParam_NumArms)),
      num_joints_(setting.get(PlannerIntParam_NumJoints)),
      body_mass_(setting.get(PlannerDoubleParam_BodyMass)), 
      body_inertia_(setting.get(PlannerMatrixParam_BodyInertia)), 
      hip_rel_pos_(setting.get(PlannerMatrixParam_HipRelPos)), 
      base_name_(setting.get(PlannerStringParam_BaseName)),
      leg_names_(setting.get(PlannerStringVectorParam_LegNames)),
      arm_names_(setting.get(PlannerStringVectorParam_ArmNames)),
      leg_endeff_names_(setting.get(PlannerStringVectorParam_LegEndEffNames)),
      arm_endeff_names_(setting.get(PlannerStringVectorParam_ArmEndEffNames)),
      clearance_ratio_(setting.get(PlannerDoubleParam_ClearanceRatio)),
      fric_coef_(setting.get(PlannerDoubleParam_FricCoef)),
      foot_offset_(setting.get(PlannerDoubleParam_FootOffset)),
      foot_delta_x_limit_(setting.get(PlannerDoubleParam_FootDeltaxLimit)),
      foot_delta_y_limit_(setting.get(PlannerDoubleParam_FootDeltayLimit)),
      body_kp_(setting.get(PlannerVectorParam_BodyKp)),
      body_kd_(setting.get(PlannerVectorParam_BodyKd)),
      body_max_acc_(setting.get(PlannerVectorParam_BodyMaxAcc)),
      raibert_kp_(setting.get(PlannerVectorParam_RaibertKp)), 
      body_init_pos_(setting.get(PlannerVectorParam_BodyInitPos)),
      body_init_vel_(setting.get(PlannerVectorParam_BodyInitVel)),
      joint_init_pos_(setting.get(PlannerVectorParam_JointInitPos)),
      joint_init_vel_(setting.get(PlannerVectorParam_JointInitVel)),
      foot_init_rel_pos_(setting.get(PlannerMatrixParam_FootInitRelPos)),
      gripper_init_rel_pos_(setting.get(PlannerMatrixParam_GripperInitRelPos)),
      optimizer_(body_mass_, body_inertia_, fric_coef_)
{
    for (int i = 0; i < num_legs_; i++) {
        leg_endff_name_id_map_[leg_endeff_names_[i]] = i;
    }
    for (int i = 0; i < num_arms_; i++) {
        arm_endff_name_id_map_[arm_endeff_names_[i]] = i;
    }

    ref_body_pos_ = body_init_pos_.head(3);
    ref_body_quat_ = Quaterniond(body_init_pos_(6),  // w
                                 body_init_pos_(3),  // x
                                 body_init_pos_(4),  // y
                                 body_init_pos_(5)); // z 
    ref_body_orn_ = ref_body_quat_.toRotationMatrix();
    ref_body_rpy_ = matrixToRpy(ref_body_orn_);  // Z-Y-X
    ref_body_lin_vel_wrd_ = ref_body_orn_ * body_init_vel_.head(3);
    ref_body_ang_vel_wrd_ = ref_body_orn_ * body_init_vel_.tail(3);

    foot_trajs_.resize(num_legs_);
    for (int i = 0; i < num_legs_; i++) {
        foot_trajs_[i] = FootSwingTrajectory();
        foot_trajs_[i].setStartPosition(ref_body_pos_ + ref_body_orn_ * foot_init_rel_pos_.row(i).transpose());
        foot_trajs_[i].setHeight(clearance_ratio_ * ref_body_pos_(2));
        foot_trajs_[i].setEndPosition(ref_body_pos_ + ref_body_orn_ * foot_init_rel_pos_.row(i).transpose());
    }
    foot_refs_.resize(num_legs_);
    for (int i = 0; i < num_legs_; i++) {
        foot_refs_[i].position = ref_body_pos_ + ref_body_orn_ * foot_init_rel_pos_.row(i).transpose();
        foot_refs_[i].orientation = Matrix3d::Identity();
        foot_refs_[i].linear_velocity = Vector3d::Zero();
        foot_refs_[i].angular_velocity = Vector3d::Zero();
        foot_refs_[i].force = Vector3d::Zero();
        foot_refs_[i].torque = Vector3d::Zero();
    }

    gripper_refs_.resize(num_arms_);
    for (int i = 0; i < num_arms_; i++) {
        gripper_refs_[i].position = ref_body_pos_ + ref_body_orn_ * gripper_init_rel_pos_.row(i).transpose();
        gripper_refs_[i].orientation = Matrix3d::Identity();
        gripper_refs_[i].linear_velocity = Vector3d::Zero();
        gripper_refs_[i].angular_velocity = Vector3d::Zero();
        gripper_refs_[i].force = Vector3d::Zero();
        gripper_refs_[i].torque = Vector3d::Zero();
    }

    VectorXd acc_weights;
    acc_weights.resize(6);
    acc_weights << 1., 1., 1., 10., 10., 1.;
    VectorXd frc_weights;
    frc_weights.resize(num_legs_*3);
    frc_weights.setConstant(1e-4);

    optimizer_.setAcceleraionWeights(acc_weights);
    optimizer_.setContactForceWeights(frc_weights);

    cur_q_.resize(6);
    cur_q_.setZero();
    cur_dq_.resize(6);
    cur_dq_.setZero();
    des_q_.resize(6);
    des_q_.setZero();
    des_dq_.resize(6);
    des_dq_.setZero();
    des_ddq_.resize(6);
    des_ddq_.setZero();
    rel_feet_pos_.resize(num_legs_, 3);
    rel_feet_pos_.setZero();
}

void MotionPlanner::setDesiredBodyLinearVelocity(ConstRefVector3d des_body_lin_vel) 
{
    des_body_lin_vel_lcl_ = des_body_lin_vel;
}

void MotionPlanner::setDesiredBodyAngularVelocity(ConstRefVector3d des_body_ang_vel)
{
    des_body_ang_vel_lcl_ = des_body_ang_vel;
}

void MotionPlanner::setDesiredGripperLinearVelocity(ConstRefVector3d des_gripper_lin_vel)
{
    des_gripper_lin_vel_lcl_ = des_gripper_lin_vel;
}

void MotionPlanner::setDesiredGripperAngularVelocity(ConstRefVector3d des_gripper_ang_vel)
{
    des_gripper_ang_vel_lcl_ = des_gripper_ang_vel;
}

void MotionPlanner::computeTaskReferences(ConstRefVectorXd base_pos, 
                                          ConstRefVectorXd base_vel, 
                                          quad_gait::GaitData& gait_data)
{
    // get current body state
    cur_body_pos_ = base_pos.head(3);
    cur_body_quat_ = Quaterniond(base_pos(6), base_pos(3), base_pos(4), base_pos(5));
    cur_body_orn_ = cur_body_quat_.toRotationMatrix();
    cur_body_rpy_ = matrixToRpy(cur_body_orn_);  // Z-Y-X
    cur_body_lin_vel_lcl_ = base_vel.head(3);
    cur_body_ang_vel_lcl_ = base_vel.tail(3);
    cur_body_lin_vel_wrd_ = cur_body_orn_ * cur_body_lin_vel_lcl_;
    cur_body_ang_vel_wrd_ = cur_body_orn_ * cur_body_ang_vel_lcl_;

    double yaw = cur_body_rpy_(2);
    cur_body_orn_z_ = AngleAxisd(yaw, Vector3d::UnitZ()).toRotationMatrix();

    // compute body pose and velocity reference
    ref_body_lin_vel_wrd_ = ref_body_orn_ * des_body_lin_vel_lcl_;
    ref_body_ang_vel_wrd_ = ref_body_orn_ * des_body_ang_vel_lcl_;
    ref_body_pos_ += timestep_ * ref_body_lin_vel_wrd_;
    ref_body_orn_ = ExpSO3(timestep_*ref_body_ang_vel_wrd_) * ref_body_orn_;
    ref_body_quat_ = Quaterniond(ref_body_orn_);
    ref_body_rpy_ = matrixToRpy(ref_body_orn_);

    // record the start positions of swing feet
    for (int i = 0; i < num_legs_; i++) {
        if (gait_data.liftoff_scheduled(i) == 1) {
            foot_trajs_[i].setHeight(clearance_ratio_ * ref_body_pos_(2));
            foot_trajs_[i].setStartPosition(foot_refs_[i].position);
        }
    }

    // compute swing foot poses
    for (int i = 0; i < num_legs_; i++) {
        if (gait_data.contact_state_scheduled(i) == 0) {  // in swing
            Vector3d twisting_vec(-hip_rel_pos_.row(i).transpose()(1), hip_rel_pos_.row(i).transpose()(0), 0.);
            Vector3d cur_hip_lin_vel_lcl = cur_body_lin_vel_lcl_ + cur_body_ang_vel_lcl_(2) * twisting_vec;
            Vector3d des_hip_lin_vel_lcl = des_body_lin_vel_lcl_ + des_body_ang_vel_lcl_(2) * twisting_vec;
            double rel_pos_x = cur_hip_lin_vel_lcl(0) * gait_data.time_stance(i) / 2
                               + raibert_kp_(0) * (cur_hip_lin_vel_lcl(0) - des_hip_lin_vel_lcl(0))
                               + (0.5*cur_body_pos_[2]/9.81) * (cur_body_lin_vel_lcl_(1)*des_body_ang_vel_lcl_(2));
            double rel_pos_y = cur_hip_lin_vel_lcl(1) * gait_data.time_stance(i) / 2
                               + raibert_kp_(1) * (cur_hip_lin_vel_lcl(1) - des_hip_lin_vel_lcl(1))
                               + (0.5*cur_body_pos_[2]/9.81) * (cur_body_lin_vel_lcl_(0)*des_body_ang_vel_lcl_(2));
            rel_pos_x = std::min(std::max(rel_pos_x, -foot_delta_x_limit_), foot_delta_x_limit_);
            rel_pos_y = std::min(std::max(rel_pos_y, -foot_delta_y_limit_), foot_delta_y_limit_);

            Vector3d des_foot_rel_pos(rel_pos_x, rel_pos_y, -cur_body_pos_(2));
            des_foot_rel_pos += hip_rel_pos_.row(i).transpose();
            Vector3d des_leg_endeff_pos = cur_body_pos_ + cur_body_orn_z_*des_foot_rel_pos;
            des_leg_endeff_pos(2) = foot_offset_;

            foot_trajs_[i].setEndPosition(des_leg_endeff_pos);
            foot_trajs_[i].computeSwingTrajectory(gait_data.phase_swing(i), gait_data.time_swing(i));

            foot_refs_[i].position = foot_trajs_[i].getPosition();
            foot_refs_[i].linear_velocity = foot_trajs_[i].getVelocity();
        }
    }

    // compute stance foot wrenches
    // actual body q and dq
    cur_q_.head(3) = cur_body_pos_;
    cur_q_.tail(3) = cur_body_rpy_;
    cur_dq_.head(3) = cur_body_lin_vel_wrd_;
    cur_dq_.tail(3) = cur_body_ang_vel_lcl_;
    // desired body q and dq
    des_q_.head(3) = ref_body_pos_;
    des_q_.tail(3) = ref_body_rpy_;
    des_dq_.head(3) = ref_body_lin_vel_wrd_;
    des_dq_.tail(3) = des_body_ang_vel_lcl_;
    // desired ddq
    des_ddq_ = body_kp_.cwiseProduct(des_q_ - cur_q_) + body_kd_.cwiseProduct(des_dq_ - cur_dq_);
    des_ddq_ = des_ddq_.cwiseMax(-body_max_acc_).cwiseMin(body_max_acc_);  // clip

    for (int i = 0; i < num_legs_; i++) {
        rel_feet_pos_.row(i) = foot_trajs_[i].getPosition() - cur_body_pos_;
    }

    const MatrixXd& des_cont_frcs = optimizer_.computeContactForces(
        yaw, rel_feet_pos_, des_ddq_, 
        gait_data.contact_state_scheduled
    );

    for (int i = 0; i < num_legs_; i++) {
        foot_refs_[i].force = des_cont_frcs.row(i).transpose();
    }

    // compute gripper poses
    for (int i= 0; i < num_arms_; i++) {
        gripper_refs_[i].linear_velocity = gripper_refs_[i].orientation * des_gripper_lin_vel_lcl_;
        gripper_refs_[i].angular_velocity = gripper_refs_[i].orientation * des_gripper_ang_vel_lcl_;
        gripper_refs_[i].position = gripper_refs_[i].position + gripper_refs_[i].linear_velocity * timestep_;
        gripper_refs_[i].orientation = ExpSO3(timestep_*gripper_refs_[i].angular_velocity) * gripper_refs_[i].orientation;
        ref_gripper_rpy_ = matrixToRpy(gripper_refs_[i].orientation);
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getBodyPositionReference() const { return ref_body_pos_; }
const MotionPlanner::Matrix3d& MotionPlanner::getBodyOrientationReference() const { return ref_body_orn_; }
const MotionPlanner::Vector3d& MotionPlanner::getBodyEulerRPYReference() const { return ref_body_rpy_; }
const MotionPlanner::Vector3d& MotionPlanner::getBodyLinearVelocityReference() const { return ref_body_lin_vel_wrd_; }
const MotionPlanner::Vector3d& MotionPlanner::getBodyAngularVelocityReference() const { return ref_body_ang_vel_wrd_; }
const MotionPlanner::Vector3d& MotionPlanner::getBodyEulerRPYRateReference() const { return des_body_ang_vel_lcl_; }

const MotionPlanner::Vector3d& MotionPlanner::getFootPositionReference(const std::string& foot_name) const
{
    auto it = leg_endff_name_id_map_.find(foot_name);
    if (it != leg_endff_name_id_map_.end()) {
        return foot_refs_[it->second].position;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getFootPositionReference] foot name can not be found!");
    }
}

const MotionPlanner::Matrix3d& MotionPlanner::getFootOrientationReference(const std::string& foot_name) const
{
    auto it = leg_endff_name_id_map_.find(foot_name);
    if (it != leg_endff_name_id_map_.end()) {
        return foot_refs_[it->second].orientation;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getFootOrientationReference] foot name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getFootLinearVelocityReference(const std::string& foot_name) const
{
    auto it = leg_endff_name_id_map_.find(foot_name);
    if (it != leg_endff_name_id_map_.end()) {
        return foot_refs_[it->second].linear_velocity;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getFootLinearVelocityReference] foot name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getFootAngularVelocityReference(const std::string& foot_name) const
{
    auto it = leg_endff_name_id_map_.find(foot_name);
    if (it != leg_endff_name_id_map_.end()) {
        return foot_refs_[it->second].angular_velocity;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getFootAngularVelocityReference] foot name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getFootForceReference(const std::string& foot_name) const
{
    auto it = leg_endff_name_id_map_.find(foot_name);
    if (it != leg_endff_name_id_map_.end()) {
        return foot_refs_[it->second].force;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getFootForceReference] foot name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getFootTorqueReference(const std::string& foot_name) const
{
    auto it = leg_endff_name_id_map_.find(foot_name);
    if (it != leg_endff_name_id_map_.end()) {
        return foot_refs_[it->second].torque;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getFootTorqueReference] foot name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getGripperPositionReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return gripper_refs_[it->second].position;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperPositionReference] gripper name can not be found!");
    }
}

const MotionPlanner::Matrix3d& MotionPlanner::getGripperOrientationReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return gripper_refs_[it->second].orientation;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperOrientationReference] gripper name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getGripperEulerRPYReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return ref_gripper_rpy_;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperEulerRPYReference] gripper name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getGripperLinearVelocityReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return gripper_refs_[it->second].linear_velocity;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperLinearVelocityReference] gripper name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getGripperAngularVelocityReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return gripper_refs_[it->second].angular_velocity;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperAngularVelocityReference] gripper name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getGripperEulerRPYRateReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return des_gripper_ang_vel_lcl_;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperEulerRPYRateReference] gripper name can not be found!");
    }
}

const MotionPlanner::Vector3d& MotionPlanner::getGripperForceReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return gripper_refs_[it->second].force;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperForceReference] gripper name can not be found!");
    }
}
const MotionPlanner::Vector3d& MotionPlanner::getGripperTorqueReference(const std::string& gripper_name) const
{
    auto it = arm_endff_name_id_map_.find(gripper_name);
    if (it != arm_endff_name_id_map_.end()) {
        return gripper_refs_[it->second].torque;
    }
    else {
        throw std::runtime_error("[legarm_mp/MotionPlanner::getGripperTorqueReference] gripper name can not be found!");
    }
}

}  // end legarm_mp namespace