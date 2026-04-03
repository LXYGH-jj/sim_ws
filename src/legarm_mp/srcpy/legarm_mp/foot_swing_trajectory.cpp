/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   foot_swing_trajectory.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Python binding for FootSwingTrajectory class
 *  @date   July 30, 2022
 **/

#include "legarm_mp/foot_swing_trajectory.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace legarm_mp;
namespace py = pybind11;

void bind_foot_swing_trajectory(py::module& m)
{
    py::class_<FootSwingTrajectory>(m, "FootSwingTrajectory")
        .def(py::init<>())
        .def("setStartPosition", &FootSwingTrajectory::setStartPosition)
        .def("setEndPosition", &FootSwingTrajectory::setEndPosition)
        .def("setHeight", &FootSwingTrajectory::setHeight)
        .def("computeSwingTrajectory", &FootSwingTrajectory::computeSwingTrajectory)
        .def("getPosition", &FootSwingTrajectory::getPosition)
        .def("getVelocity", &FootSwingTrajectory::getVelocity)
        .def("getAcceleration", &FootSwingTrajectory::getAcceleration)
        ;
}