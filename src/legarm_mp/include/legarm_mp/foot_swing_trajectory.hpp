/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   foot_swing_trajectory.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Header file for FootSwingTrajectory class
 *  @date   July 30, 2022
 **/

#pragma once

#include <Eigen/Dense>

namespace legarm_mp {

class FootSwingTrajectory
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::Vector3d Vector3d;
    typedef const Eigen::Ref<const Vector3d>&  ConstRefVector3d;

    FootSwingTrajectory();

    // Set the start position of the foot
    void setStartPosition(ConstRefVector3d start_pos);
    // Set the desired final position of the foot
    void setEndPosition(ConstRefVector3d end_pos);
    // Set the maximum height of the swing
    void setHeight(const double height);

    /**
     * @brief Compute foot swing trajectory with a bazier curve
     * 
     * @param input_phase How far along we are in the swing (0 to 1)
     * @param swing_time How long the swing should take (seconds)
     */
    void computeSwingTrajectory(const double input_phase, const double swing_time);

    // Get interpolated position
    const Vector3d& getPosition() const;
    // Get interpolated velocity
    const Vector3d& getVelocity() const;
    // Get interpolated acceleration
    const Vector3d& getAcceleration() const;

private:
    Vector3d start_pos_, end_pos_;
    double height_;
    Vector3d interp_pos_, interp_vel_, interp_acc_;
};

}  // end legarm_mp namespace