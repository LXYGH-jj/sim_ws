/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   qp_force_optimizer.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Source file for QPForceOptimizer class
 *  @date   July 31, 2022
 **/

#include "legarm_mp/qp_force_optimizer.hpp"

namespace legarm_mp {

namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    // Convert vector to skew-symmetric matrix
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    M << 0, -v[2], v[1],
         v[2], 0, -v[0], 
        -v[1], v[0], 0;
    return M;
}

}  // end namespace


QPForceOptimizer::QPForceOptimizer(double body_mass, ConstRefMatrix3d body_inertia, double fric_coef)
    : body_mass_(body_mass), 
      body_inertia_(body_inertia),
      fric_coef_(fric_coef), 
      fz_min_ratio_(0.1),
      fz_max_ratio_(10.), 
      g_(9.81)
{
    body_inertia_inv_ = body_inertia_.inverse();

    M_.resize(6, 3*kNumLegs);
    M_.setZero();
    for (int i = 0; i < kNumLegs; i++) {
        M_.block<3,3>(0, 3*i) = Matrix3d::Identity() / body_mass_;
    }

    g_bar_.resize(6);
    g_bar_ << 0., 0., -g_, 0., 0., 0.;

    Q_.resize(6, 6);
    Q_.setIdentity();
    R_.resize(3*kNumLegs, 3*kNumLegs);
    R_.setIdentity();

    Eigen::Matrix<double,5,3> fric_cone_mat;
    fric_cone_mat <<  1.,  0.,  -fric_coef_, 
                     -1.,  0.,  -fric_coef_,
                      0.,  1.,  -fric_coef_,
                      0., -1.,  -fric_coef_,
                      0.,  0.,  -1.,

    H_.resize(5*kNumLegs, 3*kNumLegs);
    H_.setZero();
    for (int i = 0; i < kNumLegs; i++) {
        H_.block<5,3>(5*i,3*i) = fric_cone_mat;
    }

    D_.resize(kNumLegs, 3*kNumLegs);
    D_.setZero();
    for (int i = 0; i < kNumLegs; i++) {
        D_(i, 3*i+2) = 1;
    }

    Rz_.setIdentity();

    P_.resize(3*kNumLegs, 3*kNumLegs);
    P_.setZero();
    q_.resize(3*kNumLegs);
    q_.setZero();
    A_.resize(5*kNumLegs+kNumLegs, 3*kNumLegs);
    A_.topRows(5*kNumLegs) = H_;
    A_.bottomRows(kNumLegs) = D_;

    l_.resize(5*kNumLegs+kNumLegs);
    l_.topRows(5*kNumLegs) = -OsqpEigen::INFTY * Eigen::Matrix<double, 5*kNumLegs, 1>::Ones();
    l_.bottomRows(kNumLegs) = Eigen::Matrix<double, kNumLegs, 1>::Zero();

    u_.resize(5*kNumLegs+kNumLegs);
    u_.setZero();

    // setting the initial data of the QP solver
    solver_.settings()->setVerbosity(false);
    solver_.settings()->setWarmStart(true);
    solver_.settings()->setPolish(true);
    solver_.settings()->setAdaptiveRhoInterval(25);
    solver_.settings()->setAbsoluteTolerance(1e-3);
    solver_.settings()->setRelativeTolerance(1e-3);

    solver_.data()->setNumberOfVariables(3*kNumLegs);
    solver_.data()->setNumberOfConstraints(5*kNumLegs+kNumLegs);

    des_lam_.resize(kNumLegs, 3);
    des_lam_.setZero();
}

void QPForceOptimizer::setAcceleraionWeights(ConstRefVectorXd acc_weights)
{
    assert(acc_weights.size() == 6);
    Q_ = acc_weights.asDiagonal();
}

void QPForceOptimizer::setContactForceWeights(ConstRefVectorXd frc_weights)
{
    assert(frc_weights.size() == 3*kNumLegs);
    R_ = frc_weights.asDiagonal();
}

const QPForceOptimizer::MatrixXd& QPForceOptimizer::computeContactForces(double yaw_angle, 
                                                                         ConstRefMatrixXd rel_feet_pos, 
                                                                         ConstRefVectorXd des_body_acc, 
                                                                         ConstRefVectorXi cont_states)
{
    Rz_ = Eigen::AngleAxisd(yaw_angle, Vector3d::UnitZ()).toRotationMatrix();

    for (int i = 0; i < kNumLegs; i++) {
        M_.block<3,3>(3, 3*i) = body_inertia_inv_ * Rz_.transpose() * skew(rel_feet_pos.row(i));
        if (cont_states(i) == 0) {  // in swing
            l_(5*kNumLegs+i) = - 1e-7;
            u_(5*kNumLegs+i) = + 1e-7;
        }
        else if (cont_states(i) == 1) {  // in stance
            l_(5*kNumLegs+i) = fz_min_ratio_ * body_mass_ * g_;
            u_(5*kNumLegs+i) = fz_max_ratio_ * body_mass_ * g_;
        }
        else {
            throw std::runtime_error("unknown contact state.");
        }
    }

    P_ = M_.transpose() * Q_ * M_ + R_;
    q_ = (g_bar_ - des_body_acc).transpose() * Q_ * M_;

    Eigen::SparseMatrix<double> hessian_matrix = P_.sparseView();
    Eigen::VectorXd gradient = q_;
    Eigen::SparseMatrix<double> linear_constraints_matrix = A_.sparseView();
    Eigen::VectorXd lower_bound = l_;
    Eigen::VectorXd upper_bound = u_;

    if (!solver_.isInitialized()) {
        // set the initial data of the QP solver
        if (!solver_.data()->setHessianMatrix(hessian_matrix))
            throw "hessian matrix cannot be set correctly.";
        if (!solver_.data()->setGradient(gradient))
            throw "gradient cannot be set correctly.";
        if (!solver_.data()->setLinearConstraintsMatrix(linear_constraints_matrix))
            throw "linear constraints matrix cannot be set correctly.";
        if (!solver_.data()->setBounds(lower_bound, upper_bound))
            throw "lower and upper bound cannot be set correctly.";

        // instantiate the solver
        if (!solver_.initSolver())
            throw "solver cannot be initialized correctly.";
    }
    else {
        // update the data of the QP solver
        if (!solver_.updateHessianMatrix(hessian_matrix))
            throw "hessian matrix cannot be updated correctly.";
        if (!solver_.updateGradient(gradient))
            throw "gradient cannot be updated correctly.";
        if (!solver_.updateLinearConstraintsMatrix(linear_constraints_matrix))
            throw "linear constraints matrix cannot be updated correctly.";
        if (!solver_.updateBounds(lower_bound, upper_bound))
            throw "lower and upper bound cannot be updated correctly.";
    }

    // solve the QP problem
    if(solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
        throw "QP problem cannot be solved.";

    const VectorXd& solution = solver_.getSolution();

    for (int i = 0; i < kNumLegs; i++) {
        des_lam_.row(i) = solution.segment<3>(3*i);
    }

    return des_lam_;
}

}  // end legarm_mp namespace