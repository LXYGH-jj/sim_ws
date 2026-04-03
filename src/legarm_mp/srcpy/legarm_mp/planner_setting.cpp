/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   module.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Python binding for PlannerSetting class
 *  @date   July 26, 2022
 **/

#include "legarm_mp/planner_setting.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace legarm_mp;
namespace py = pybind11;

void bind_planner_setting(py::module& m)
{
    py::class_<PlannerSetting>(m, "PlannerSetting")
        .def(py::init<>())
        .def("initialize", &PlannerSetting::initialize, 
            py::arg("rootdir"), py::arg("cfg_file"), py::arg("planner_vars_yaml")="planner_variables")
        .def("get", (const std::string& (PlannerSetting::*)(PlannerStringParam) const) &PlannerSetting::get)
        .def("get", (const std::vector<std::string>& (PlannerSetting::*)(PlannerStringVectorParam) const) &PlannerSetting::get)
        .def("get", (int (PlannerSetting::*)(PlannerIntParam) const) &PlannerSetting::get)
        .def("get", (double (PlannerSetting::*)(PlannerDoubleParam) const) &PlannerSetting::get)
        .def("get", (const Eigen::MatrixXd& (PlannerSetting::*)(PlannerMatrixParam) const) &PlannerSetting::get)
        ;
}