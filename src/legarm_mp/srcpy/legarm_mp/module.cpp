/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   module.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Python binding for the Motion Planner for the Legged Mobile Manipulator
 *  @date   July 26, 2022
 **/

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_planner_params(py::module& module);
void bind_planner_setting(py::module& module);
void bind_foot_swing_trajectory(py::module& module);
void bind_qp_force_optimizer(py::module& module);
void bind_motion_planner(py::module& module);

PYBIND11_MODULE(legarm_mp_pywrap, m)
{
    m.doc() = R"pbdoc(
        Python Bindings of the Motion Planner for the Legged Mobile Manipulator.
        ---------------------------------------
        .. currentmodule: legarm_mp_pywrap
        .. autosummry::
           : toctree: _generate
    )pbdoc";

    bind_planner_params(m);
    bind_planner_setting(m);
    bind_foot_swing_trajectory(m);
    bind_qp_force_optimizer(m);
    bind_motion_planner(m);

    #ifdef VERSION_INFO
      m.attr("__version__") = VERSION_INFO;
    #else
      m.attr("__version__") = "dev";
    #endif
}