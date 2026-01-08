/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   test_csv_writer.cpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Test file for CSVWriter class
 *  @date   July 08, 2022
 **/

#include "commutils/csv/csv_writer.hpp"

#include <gtest/gtest.h>

class CSVWriterTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CSVWriterTest, TestingCSVWriter)
{
    using namespace CSV;

    CSVWriter writer("data.csv");
    std::vector<std::string> data_headings = {
        "time", "data"
    };
    writer.addDataHeading(data_headings);

    Eigen::VectorXd data;
    data.resize(data_headings.size());
    data << 0.0, 1.0;
    writer.addDataInRow(data);
}