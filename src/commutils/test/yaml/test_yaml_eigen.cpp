/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   test_yaml_eigen.cpp
 * @author Jun Li (junlileeds@gmail.com)
 * @brief  Test file for yaml-eigen functions
 * @date   2021-11-12
 */

#include "commutils/yaml/yaml_cpp_fwd.hpp"

#include <gtest/gtest.h>

class YamlEigenTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(YamlEigenTest, TestingReadEigenParameterFunction)
{    
    Eigen::Vector4d vector_variable_parameter;
    Eigen::Vector4d value_vector_variable_parameter(1, 3, 0, 2);
    Eigen::Vector4d value_vector_default(0, 0, 0, 0);

    Eigen::Matrix3d matrix_variable_parameter;
    Eigen::Matrix3d value_matrix_variable_parameter;
    value_matrix_variable_parameter << 1, 2, 3, 
                                       4, 5, 6, 
                                       7, 8, 9;
    Eigen::Matrix3d value_matrix_default;
    value_matrix_default << 0, 0, 0,
                            0, 0, 0,
                            0, 0, 0;

    Eigen::Quaterniond quaternion_variable_parameter;
    Eigen::Quaterniond value_quaternion_variable_parameter(1, 0, 0, 0);
    Eigen::Quaterniond value_quaternion_default(1, 0, 0, 0);

    YAML::Node with_variable_node, without_variable_node;
    with_variable_node["vector"] = value_vector_variable_parameter;
    with_variable_node["matrix"] = value_matrix_variable_parameter;
    with_variable_node["quaternion"] = value_quaternion_variable_parameter;

    // Optional and variable exists
    YAML::readParameter(with_variable_node, "vector", 
        vector_variable_parameter, true);
    ASSERT_EQ(value_vector_variable_parameter, vector_variable_parameter);
    YAML::readParameter(with_variable_node, "matrix", 
        matrix_variable_parameter, true);
    ASSERT_EQ(value_matrix_variable_parameter, matrix_variable_parameter);
    YAML::readParameter(with_variable_node, "quaternion", 
        quaternion_variable_parameter, true);
    ASSERT_EQ(value_quaternion_variable_parameter.coeffs(), 
        quaternion_variable_parameter.coeffs());

    // Mandatory and variable exists
    YAML::readParameter(with_variable_node, "vector", 
        vector_variable_parameter, false);
    ASSERT_EQ(value_vector_variable_parameter, vector_variable_parameter);
    YAML::readParameter(with_variable_node, "matrix", 
        matrix_variable_parameter, false);
    ASSERT_EQ(value_matrix_variable_parameter, matrix_variable_parameter);
    YAML::readParameter(with_variable_node, "quaternion", 
        quaternion_variable_parameter, false);
    ASSERT_EQ(value_quaternion_variable_parameter.coeffs(), 
        quaternion_variable_parameter.coeffs());


    // Optional and variable does not exists
    vector_variable_parameter = Eigen::Vector4d(0, 0, 0, 0);
    YAML::readParameter(without_variable_node, "vector", 
        vector_variable_parameter, true);
    ASSERT_EQ(Eigen::Vector4d(0, 0, 0, 0), vector_variable_parameter);
    matrix_variable_parameter << 0, 0, 0, 0, 0, 0, 0, 0, 0;
    YAML::readParameter(without_variable_node, "matrix", 
        matrix_variable_parameter, true);
    Eigen::Matrix3d reference_matrix_variable_parameter;
    reference_matrix_variable_parameter << 0, 0, 0, 0, 0, 0, 0, 0, 0;
    ASSERT_EQ(reference_matrix_variable_parameter, matrix_variable_parameter);
    quaternion_variable_parameter = Eigen::Quaterniond(0, 0, 0, 0);
    YAML::readParameter(without_variable_node, "quaternion", 
        quaternion_variable_parameter, true);
    ASSERT_EQ(Eigen::Quaterniond(0, 0, 0, 0).coeffs(), 
        quaternion_variable_parameter.coeffs());

    // Mandatory and variable does not exists
    try
    {
        YAML::readParameter(without_variable_node, "vector", 
            vector_variable_parameter, false);
    }
    catch(const std::runtime_error& error)
    {
        std::string e_str = error.what();
        assert(e_str.compare("Error reading the yaml parameter [vector]") == 0);
    }

    try
    {
        YAML::readParameter(without_variable_node, "matrix", 
            matrix_variable_parameter, false);
    }
    catch(const std::runtime_error& error)
    {
        std::string e_str = error.what();
        assert(e_str.compare("Error reading the yaml parameter [matrix]") == 0);
    }

    try
    {
        YAML::readParameter(without_variable_node, "quaternion", 
            quaternion_variable_parameter, false);
    }
    catch(const std::runtime_error& error)
    {
        std::string e_str = error.what();
        assert(e_str.compare("Error reading the yaml parameter [quaternion]") == 0);
    }
    
    // Default value: Use provided value if exists.
    YAML::readParameterDefault(with_variable_node, "vector", 
        vector_variable_parameter, value_vector_default);
    ASSERT_EQ(value_vector_variable_parameter, vector_variable_parameter);
    YAML::readParameterDefault(without_variable_node, "vector", 
        vector_variable_parameter, value_vector_default);
    ASSERT_EQ(value_vector_default, vector_variable_parameter);
    YAML::readParameterDefault(with_variable_node, "matrix", 
        matrix_variable_parameter, value_matrix_default);
    ASSERT_EQ(value_matrix_variable_parameter, matrix_variable_parameter);
    YAML::readParameterDefault(without_variable_node, "matrix", 
        matrix_variable_parameter, value_matrix_default);
    ASSERT_EQ(value_matrix_default, matrix_variable_parameter);
    YAML::readParameterDefault(with_variable_node, "quaternion", 
        quaternion_variable_parameter, value_quaternion_default);
    ASSERT_EQ(value_quaternion_variable_parameter.coeffs(), 
        quaternion_variable_parameter.coeffs());
    YAML::readParameterDefault(without_variable_node, "quaternion", 
        quaternion_variable_parameter, value_quaternion_default);
    ASSERT_EQ(value_quaternion_default.coeffs(), 
        quaternion_variable_parameter.coeffs());
}