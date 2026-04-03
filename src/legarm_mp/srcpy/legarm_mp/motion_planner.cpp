/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   motion_planner.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Python binding for MotionPlanner class
 *  @date   August 05, 2022
 **/

#include "legarm_mp/motion_planner.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using namespace legarm_mp;
namespace py = pybind11;

void bind_motion_planner(py::module& m)
{
    py::class_<LinkStateRef>(m, "LinkStateRef")
        .def(py::init<>())
        .def_readwrite("position", &LinkStateRef::position)
        .def_readwrite("orientation", &LinkStateRef::orientation)
        .def_readwrite("linear_velocity", &LinkStateRef::linear_velocity)
        .def_readwrite("angular_velocity", &LinkStateRef::angular_velocity)
        .def_readwrite("force", &LinkStateRef::force)
        .def_readwrite("torque", &LinkStateRef::torque)
        ;

    py::class_<MotionPlanner>(m, "MotionPlanner")
        .def(py::init<const PlannerSetting&>())
        .def("setDesiredBodyLinearVelocity", &MotionPlanner::setDesiredBodyLinearVelocity)
        .def("setDesiredBodyAngularVelocity", &MotionPlanner::setDesiredBodyAngularVelocity)
        .def("setDesiredGripperLinearVelocity", &MotionPlanner::setDesiredGripperLinearVelocity)
        .def("setDesiredGripperAngularVelocity", &MotionPlanner::setDesiredGripperAngularVelocity)
        .def("computeTaskReferences", &MotionPlanner::computeTaskReferences)
        .def("getBodyPositionReference", &MotionPlanner::getBodyPositionReference)
        .def("getBodyOrientationReference", &MotionPlanner::getBodyOrientationReference)
        .def("getBodyEulerRPYReference", &MotionPlanner::getBodyEulerRPYReference)
        .def("getBodyLinearVelocityReference", &MotionPlanner::getBodyLinearVelocityReference)
        .def("getBodyAngularVelocityReference", &MotionPlanner::getBodyAngularVelocityReference)
        .def("getBodyEulerRPYRateReference", &MotionPlanner::getBodyEulerRPYRateReference)
        .def("getFootPositionReference", &MotionPlanner::getFootPositionReference)
        .def("getFootOrientationReference", &MotionPlanner::getFootOrientationReference)
        .def("getFootLinearVelocityReference", &MotionPlanner::getFootLinearVelocityReference)
        .def("getFootAngularVelocityReference", &MotionPlanner::getFootAngularVelocityReference)
        .def("getFootForceReference", &MotionPlanner::getFootForceReference)
        .def("getFootTorqueReference", &MotionPlanner::getFootTorqueReference)
        .def("getGripperPositionReference", &MotionPlanner::getGripperPositionReference)
        .def("getGripperOrientationReference", &MotionPlanner::getGripperOrientationReference)
        .def("getGripperEulerRPYReference", &MotionPlanner::getGripperEulerRPYReference)
        .def("getGripperLinearVelocityReference", &MotionPlanner::getGripperLinearVelocityReference)
        .def("getGripperAngularVelocityReference", &MotionPlanner::getGripperAngularVelocityReference)
        .def("getGripperEulerRPYRateReference", &MotionPlanner::getGripperEulerRPYRateReference)
        .def("getGripperForceReference", &MotionPlanner::getGripperForceReference)
        .def("getGripperTorqueReference", &MotionPlanner::getGripperTorqueReference)
        ;
}