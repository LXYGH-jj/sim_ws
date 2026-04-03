/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   qp_force_optimizer.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Header file for QPForceOptimizer class
 *  @date   July 30, 2022
 **/

#pragma once

#include <Eigen/Eigen>

#include <OsqpEigen/OsqpEigen.h>

namespace legarm_mp {

constexpr int kNumLegs = 4;

class QPForceOptimizer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::Vector3d Vector3d;
    typedef Eigen::Matrix3d Matrix3d;
    typedef Eigen::VectorXi VectorXi;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::MatrixXd MatrixXd;
    typedef const Eigen::Ref<const Matrix3d>&  ConstRefMatrix3d;
    typedef const Eigen::Ref<const VectorXi>&  ConstRefVectorXi;
    typedef const Eigen::Ref<const VectorXd>&  ConstRefVectorXd;
    typedef const Eigen::Ref<const MatrixXd>&  ConstRefMatrixXd;

    QPForceOptimizer(double body_mass, ConstRefMatrix3d body_inertia, double fric_coef);

    void setAcceleraionWeights(ConstRefVectorXd acc_weights);
    void setContactForceWeights(ConstRefVectorXd frc_weights);
    const MatrixXd& computeContactForces(double yaw_angle, 
                                         ConstRefMatrixXd rel_feet_pos, 
                                         ConstRefVectorXd des_body_acc,
                                         ConstRefVectorXi cont_states);

private:
    double body_mass_;  // equivalent robot mass
    Matrix3d body_inertia_;  // equivalent robot inertia
    Matrix3d body_inertia_inv_;
    double fric_coef_;
    double fz_min_ratio_, fz_max_ratio_;
    double g_;

    MatrixXd M_;
    VectorXd g_bar_;
    MatrixXd Q_;  // weight matrix
    MatrixXd R_;  // weight matrix
    MatrixXd H_;  // friction pyramid matrix
    MatrixXd D_;  // contact selection matrix

    Matrix3d Rz_; // rotation matrix

    // standard QP problem and solver
    MatrixXd P_;
    VectorXd q_;
    MatrixXd A_;
    VectorXd l_, u_;
    OsqpEigen::Solver solver_;

    MatrixXd des_lam_;  // optimal contact forces
};

}  // end legarm_mp namespace