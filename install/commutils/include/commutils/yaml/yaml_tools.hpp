/**
 * @file yaml_tools.hpp
 * @author Maximilien Naveau (maximilien.naveau@gmail.com)
 * @license License BSD-3-Clause
 * @copyright Copyright (c) 2019, New York University and Max Planck Gesellschaft.
 * @date 2019-09-30
 */

#pragma once

#include "commutils/yaml/yaml_eigen.hpp"

namespace YAML {

/**
 * @brief helper function to safely read a yaml parameter
 * 
 * @tparam YamlType
 * @param node
 * @param name
 * @return YamlType
 */
template<typename YamlType>
static YamlType readParameter(const YAML::Node& node, const std::string& name)
{
    try
    {
        return node[name.c_str()].as<YamlType>();
    }
    catch(...)
    {
        throw std::runtime_error("Error reading the yaml parameter [" + name + "]");
    }
}

/**
 * @brief helper function to safely read a yaml parameter
 * 
 * @tparam YamlType
 * @param node
 * @param name
 * @param parameter
 * @param optional
 */
template<typename YamlType>
static void readParameter(const YAML::Node& node, const std::string& name, 
        YamlType& parameter, bool optional = false)
{
    if (optional && !node[name.c_str()])
    {
        return;
    }
    parameter = readParameter<YamlType>(node, name);
}

/**
 * @brief helper function to safely read a yaml parameter with default value
 * 
 * @tparam YamlType
 * @param node
 * @param name
 * @param parameter
 * @param default_value
 */
template<typename YamlType>
static void readParameterDefault(const YAML::Node& node, const std::string& name, 
        YamlType& parameter, YamlType default_value)
{
    if (!node[name.c_str()])
    {
        parameter = default_value;
    }
    else
    {
        parameter = readParameter<YamlType>(node, name);
    }
}

}  // namespace YAML