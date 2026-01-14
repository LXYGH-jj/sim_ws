/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   csv_writer.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  File for CSVWriter class
 *  @date   July 08, 2022
 **/

#pragma once

#include <string>
#include <vector>
#include <fstream>

#include <Eigen/Dense>

namespace CSV {

class CSVWriter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef Eigen::VectorXd VectorXd;
    typedef const Eigen::Ref<const VectorXd>&  ConstRefVectorXd;

    CSVWriter(const std::string& filename, const std::string& delm = ",")
        : filename_(filename), delimeter_(delm), lines_count_(0)
    {}

    void addDataHeading(const std::vector<std::string>& heading)
    {
        std::fstream file;
        // Open the file in truncate mode if first line else in Append Mode
        file.open(filename_, std::ios::out | (lines_count_ ? std::ios::app : std::ios::trunc));
        // Iterate over the range and add each element to file seperated by delimeter.
        auto first = heading.begin();
        auto last = heading.end();
        for (; first != last; ) {
            file << *first;
            if (++first != last)
                file << delimeter_;
        }
        file << "\n";
        lines_count_++;
        // Close the file
        file.close();
    }

    void addDataInRow(ConstRefVectorXd data)
    {
        std::fstream file;
        // Open the file in truncate mode if first line else in Append Mode
        file.open(filename_, std::ios::out | (lines_count_ ? std::ios::app : std::ios::trunc));
        // Iterate over the range and add each element to file seperated by delimeter.
        std::size_t data_size = data.size();
        for (std::size_t i = 0; i < data_size; i++) {
            file << data(i);
            if (i < data_size-1)
                file << delimeter_;
        }
        file << "\n";
        lines_count_++;
        // Close the file
        file.close();
    }

private:
    std::string filename_;
    std::string delimeter_;
    int lines_count_;
};

}  // end CSV namespace