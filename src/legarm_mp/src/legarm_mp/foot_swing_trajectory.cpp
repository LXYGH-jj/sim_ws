/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   foot_swing_trajectory.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Source file for FootSwingTrajectory class
 *  @date   July 30, 2022
 **/

#include "legarm_mp/foot_swing_trajectory.hpp"

#include <cmath>

namespace legarm_mp {

namespace {

typedef Eigen::Vector3d Vector3d;
typedef const Eigen::Ref<const Vector3d>&  ConstRefVector3d;

/**
 * Generate cubic bezier interpolation between start position and end position.
 * input_phase is between [0, 1].
 */
double cubicBezier(double input_phase, double start_pos, double end_pos)
{
    assert(input_phase >= 0 && input_phase <= 1);
    double bezier = std::pow(input_phase, 3.) + 3. * (std::pow(input_phase, 2.) * (1. - input_phase));
    return start_pos + bezier * (end_pos - start_pos);
}

Vector3d cubicBezier(double input_phase, ConstRefVector3d start_pos, ConstRefVector3d end_pos)
{
    assert(input_phase >= 0 && input_phase <= 1);
    double bezier = std::pow(input_phase, 3.) + 3. * (std::pow(input_phase, 2.) * (1. - input_phase));
    return start_pos + bezier * (end_pos - start_pos);
}

/** 
 * Generate cubic bezier interpolation first derivative between start 
 * position and end position. input_phase is between [0, 1].
 */
double cubicBezierFirstDerivative(double input_phase, double start_pos, double end_pos)
{
    assert(input_phase >= 0 && input_phase <= 1);
    double bezier_first_derivative = 6. * input_phase * (1. - input_phase);
    return bezier_first_derivative * (end_pos - start_pos);
}

Vector3d cubicBezierFirstDerivative(double input_phase, ConstRefVector3d start_pos, ConstRefVector3d end_pos)
{
    assert(input_phase >= 0 && input_phase <= 1);
    double bezier_first_derivative = 6. * input_phase * (1. - input_phase);
    return bezier_first_derivative * (end_pos - start_pos);
}

/**
 * Generate cubic bezier interpolation second derivative between start 
 * position and end position. input_phase is between [0, 1].
 */
double cubicBezierSecondDerivative(double input_phase, double start_pos, double end_pos)
{
    assert(input_phase >= 0 && input_phase <= 1);
    double bezier_second_derivative = 6. - 12. * input_phase;
    return bezier_second_derivative * (end_pos - start_pos);
}

Vector3d cubicBezierSecondDerivative(double input_phase, ConstRefVector3d start_pos, ConstRefVector3d end_pos)
{
    assert(input_phase >= 0 && input_phase <= 1);
    double bezier_second_derivative = 6. - 12. * input_phase;
    return bezier_second_derivative * (end_pos - start_pos);
}

}  // end namespace

FootSwingTrajectory::FootSwingTrajectory()
{
    start_pos_.setZero();
    end_pos_.setZero();
    height_ = 0.;
    interp_pos_.setZero();
    interp_vel_.setZero();
    interp_acc_.setZero();
}

void FootSwingTrajectory::setStartPosition(ConstRefVector3d start_pos) 
{ 
    start_pos_ = start_pos; 
    interp_pos_ = start_pos;
}
void FootSwingTrajectory::setEndPosition(ConstRefVector3d end_pos) { end_pos_ = end_pos; }
void FootSwingTrajectory::setHeight(const double height) { height_ = height; }

void FootSwingTrajectory::computeSwingTrajectory(const double input_phase, const double swing_time)
{
    interp_pos_ = cubicBezier(input_phase, start_pos_, end_pos_);
    interp_vel_ = cubicBezierFirstDerivative(input_phase, start_pos_, end_pos_) / swing_time;
    interp_acc_ = cubicBezierSecondDerivative(input_phase, start_pos_, end_pos_) / (swing_time * swing_time);

    if (input_phase < 0.5) {
        interp_pos_(2) = cubicBezier(input_phase*2., start_pos_(2), start_pos_(2) + height_);
        interp_vel_(2) = cubicBezierFirstDerivative(input_phase*2., start_pos_(2), start_pos_(2) + height_) * 2. / swing_time;
        interp_acc_(2) = cubicBezierSecondDerivative(input_phase*2., start_pos_(2), start_pos_(2) + height_) * 4. / (swing_time * swing_time);
    }
    else {
        interp_pos_(2) = cubicBezier(input_phase*2.-1., start_pos_(2) + height_, end_pos_(2));
        interp_vel_(2) = cubicBezierFirstDerivative(input_phase*2.-1., start_pos_(2) + height_, end_pos_(2)) * 2. / swing_time;
        interp_acc_(2) = cubicBezierSecondDerivative(input_phase*2.-1., start_pos_(2) + height_, end_pos_(2)) * 4. / (swing_time * swing_time);
    }
}

const Vector3d& FootSwingTrajectory::getPosition() const { return interp_pos_; }
const Vector3d& FootSwingTrajectory::getVelocity() const { return interp_vel_; }
const Vector3d& FootSwingTrajectory::getAcceleration() const { return interp_acc_; }

}  // end legarm_mp namespace