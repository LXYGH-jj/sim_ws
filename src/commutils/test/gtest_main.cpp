/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file gtest_main.cpp
 * @author Jun Li (junlileeds@gmail.com)
 * @date 2021-11-12
 */

#include <string>
#include <iostream>
#include <gtest/gtest.h>

int main(int argc, char** argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}