/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   qp_force_optimizer.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Python binding for QPForceOptimizer class
 *  @date   July 31, 2022
 **/

#include "legarm_mp/qp_force_optimizer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace legarm_mp;
namespace py = pybind11;

void bind_qp_force_optimizer(py::module& m)
{
    py::class_<QPForceOptimizer>(m, "QPForceOptimizer")
        .def(py::init<double, const Eigen::Ref<const Eigen::Matrix3d>&, double>())
        .def("setAcceleraionWeights", &QPForceOptimizer::setAcceleraionWeights)
        .def("setContactForceWeights", &QPForceOptimizer::setContactForceWeights)
        .def("computeContactForces", &QPForceOptimizer::computeContactForces)
        ;
}
