/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   test_yaml_tools.cpp
 * @author Jun Li (junlileeds@gmail.com)
 * @brief  Test file for yaml tools
 * @date   2021-11-12
 */

#include "commutils/yaml/yaml_cpp_fwd.hpp"

#include <gtest/gtest.h>

#define PRECISION 1.e-4

class YamlToolsTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(YamlToolsTest, TestingReadOptionalParameterFunction)
{
    double variable_parameter;
    double value_variable_parameter1 = 12.;
    double value_variable_parameter2 = 24.;
    double value_default = 36.;

    YAML::Node with_variable_node, without_variable_node;
    with_variable_node["variable1"] = value_variable_parameter1;
    with_variable_node["variable2"] = value_variable_parameter2;

    // Optional and variable exists
    YAML::readParameter(with_variable_node, "variable1", 
        variable_parameter, true);
    EXPECT_NEAR(value_variable_parameter1, variable_parameter, PRECISION);

    // Mandatory and variable exists
    YAML::readParameter(with_variable_node, "variable2", 
        variable_parameter, false);
    EXPECT_NEAR(value_variable_parameter2, variable_parameter, PRECISION);

    // Optional and variable does not exists
    variable_parameter = 0.0;
    YAML::readParameter(without_variable_node, "variable1", 
        variable_parameter, true);
    EXPECT_NEAR(0.0, variable_parameter, PRECISION);

    // Mandatory and variable does not exists
    variable_parameter = 0.0;
    try
    {
        YAML::readParameter(without_variable_node, "variable2", 
            variable_parameter, false);
    }
    catch(const std::runtime_error& error)
    {
        std::string e_str = error.what();
        assert(e_str.compare("Error reading the yaml parameter [variable2]") == 0);
    }

    // Default value: Use provided value if exists.
    YAML::readParameterDefault(with_variable_node, "variable1", 
        variable_parameter, value_default);
    EXPECT_NEAR(value_variable_parameter1, variable_parameter, PRECISION);

    YAML::readParameterDefault(without_variable_node, "variable1", 
        variable_parameter, value_default);
    EXPECT_NEAR(value_default, variable_parameter, PRECISION);
}