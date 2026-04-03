/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   module.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Python binding for planner params
 *  @date   July 26, 2022
 **/

#include "legarm_mp/planner_params.hpp"

#include <pybind11/pybind11.h>

using namespace legarm_mp;
namespace py = pybind11;

void bind_planner_params(py::module& m)
{
    // binding string params
    py::enum_<PlannerStringParam>(m, "PlannerStringParam")
        .value("PlannerStringParam_BaseName", PlannerStringParam_BaseName)
        .export_values();

    // binding string vector params
    py::enum_<PlannerStringVectorParam>(m, "PlannerStringVectorParam")
        .value("PlannerStringVectorParam_LegNames", PlannerStringVectorParam_LegNames)
        .value("PlannerStringVectorParam_ArmNames", PlannerStringVectorParam_ArmNames)
        .value("PlannerStringVectorParam_LegEndEffNames", PlannerStringVectorParam_LegEndEffNames)
        .value("PlannerStringVectorParam_ArmEndEffNames", PlannerStringVectorParam_ArmEndEffNames)
        .export_values();

    // binding int params
    py::enum_<PlannerIntParam>(m, "PlannerIntParam")
        .value("PlannerIntParam_NumLegs", PlannerIntParam_NumLegs)
        .value("PlannerIntParam_NumArms", PlannerIntParam_NumArms)
        .value("PlannerIntParam_NumJoints", PlannerIntParam_NumJoints)
        .export_values();

    // binding double params
    py::enum_<PlannerDoubleParam>(m, "PlannerDoubleParam")
        .value("PlannerDoubleParam_TimeStep", PlannerDoubleParam_TimeStep)
        .value("PlannerDoubleParam_BodyMass", PlannerDoubleParam_BodyMass)
        .value("PlannerDoubleParam_ClearanceRatio", PlannerDoubleParam_ClearanceRatio)
        .value("PlannerDoubleParam_FricCoef", PlannerDoubleParam_FricCoef)
        .value("PlannerDoubleParam_FootOffset", PlannerDoubleParam_FootOffset)
        .value("PlannerDoubleParam_FootDeltaxLimit", PlannerDoubleParam_FootDeltaxLimit)
        .value("PlannerDoubleParam_FootDeltayLimit", PlannerDoubleParam_FootDeltayLimit)
        .export_values();

    // binding vector params
    py::enum_<PlannerVectorParam>(m, "PlannerVectorParam")
        .value("PlannerVectorParam_BodyKp", PlannerVectorParam_BodyKp)
        .value("PlannerVectorParam_BodyKd", PlannerVectorParam_BodyKd)
        .value("PlannerVectorParam_BodyMaxAcc", PlannerVectorParam_BodyMaxAcc)
        .value("PlannerVectorParam_RaibertKp", PlannerVectorParam_RaibertKp)
        .value("PlannerVectorParam_BodyInitPos", PlannerVectorParam_BodyInitPos)
        .value("PlannerVectorParam_BodyInitVel", PlannerVectorParam_BodyInitVel)
        .value("PlannerVectorParam_JointInitPos", PlannerVectorParam_JointInitPos)
        .value("PlannerVectorParam_JointInitVel", PlannerVectorParam_JointInitVel)
        .export_values();

    // binding matrix params
    py::enum_<PlannerMatrixParam>(m, "PlannerMatrixParam")
        .value("PlannerMatrixParam_BodyInertia", PlannerMatrixParam_BodyInertia)
        .value("PlannerMatrixParam_HipRelPos", PlannerMatrixParam_HipRelPos)
        .value("PlannerMatrixParam_FootInitRelPos", PlannerMatrixParam_FootInitRelPos)
        .value("PlannerMatrixParam_GripperInitRelPos", PlannerMatrixParam_GripperInitRelPos)
        .export_values();
}