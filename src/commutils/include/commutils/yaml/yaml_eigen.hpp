/**
 * @file yaml_eigen.hpp
 * @author Alexander Herzog
 * @license License BSD-3-Clause
 * @copyright Copyright (c) 2019, New York University and Max Planck Gesellschaft.
 * @date 2015-02-27
 * 
 * @brief Add support for eigen frm the yaml-cpp standard package.
 */

#pragma once

#include <type_traits>
#include <iomanip>
#include <Eigen/Eigen>
#include <yaml-cpp/yaml.h>


namespace YAML {

template<typename Scalar, int Rows, int Cols, int Align, int RowsAtCompileTime, int ColsAtCompileTime>
struct EigenVectorConverter
{
    typedef Eigen::Matrix<Scalar, Rows, Cols, Align, RowsAtCompileTime, ColsAtCompileTime> VectorType;

    static void resize_if_needed(int rows, int cols, VectorType& rhs)
    {
        if (rhs.size() != rows*cols)
        {
            if (VectorType::SizeAtCompileTime == Eigen::Dynamic && 
                VectorType::MaxSizeAtCompileTime == Eigen::Dynamic)
            {
                rhs.resize(rows, cols);
            }
            else
            {
                std::ostringstream error;
                error << "ERROR: The fixed sized vector of size (" << rhs.size()
                      << ") is of different size than the input yaml data vector of "
                      << "size (" << rows * cols << ").";
                throw(std::runtime_error(error.str()));
            }
        }
    }

    static Scalar& access_element(int row, int col, VectorType& rhs)
    {
        return rhs(row + col);
    }
};

template<typename Scalar, int Rows, int Cols, int Align, int RowsAtCompileTime, int ColsAtCompileTime>
struct EigenMatrixConverter
{
    typedef Eigen::Matrix<Scalar, Rows, Cols, Align, RowsAtCompileTime, ColsAtCompileTime> MatrixType;

    static void resize_if_needed(int rows, int cols, MatrixType& rhs)
    {
        if (rhs.rows() != rows || rhs.cols() != cols)
        {
            if (MatrixType::SizeAtCompileTime == Eigen::Dynamic && 
                MatrixType::MaxSizeAtCompileTime == Eigen::Dynamic)
            {
                rhs.resize(rows, cols);
            }
            else
            {
                std::ostringstream error;
                error << "ERROR: The fixed sized matrix of dim (" << rhs.rows() << ","
                      << rhs.cols() << ") is of different dim than the input yaml "
                      << "data matrix of dim (" << rows << "," << cols << ").";
                throw(std::runtime_error(error.str()));
            }
        }
    }

    static Scalar& access_element(int row, int col, MatrixType& rhs)
    {
        return rhs(row, col);
    }
};

template<typename Scalar, int Rows, int Cols, int Align, int RowsAtCompileTime, int ColsAtCompileTime>
struct convert<Eigen::Matrix<Scalar, Rows, Cols, Align, RowsAtCompileTime, ColsAtCompileTime>>
{
    typedef Eigen::Matrix<Scalar, Rows, Cols, Align, RowsAtCompileTime, ColsAtCompileTime> EigenType;
    const static bool is_vector_type = Rows == 1 || Cols == 1;
    typedef typename std::conditional<is_vector_type, 
        EigenVectorConverter<Scalar, Rows, Cols, Align, RowsAtCompileTime, ColsAtCompileTime>, 
        EigenMatrixConverter<Scalar, Rows, Cols, Align, RowsAtCompileTime, ColsAtCompileTime>>::type Converter;

    static Node encode(const EigenType& rhs)
    {
        Eigen::IOFormat yaml_format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ", ", "[", "]", "[", "]");
        std::stringstream ss;
        ss << std::setprecision(std::numeric_limits<double>::digits10 + 1) << rhs.format(yaml_format);
        return Load(ss.str());
    }

    static bool decode_1d(const Node& node, EigenType& rhs)
    {
        const size_t n_size = node.size();
        Converter::resize_if_needed(n_size, 1, rhs);
        for (size_t r = 0; r < n_size; r++)
            Converter::access_element(r, 0, rhs) = node[r].as<Scalar>();
        
        return true;
    }

    static bool decode_2d(const Node& node, EigenType& rhs)
    {
        const size_t n_rows = node.size();
        const size_t n_cols = node[0].size();
        Converter::resize_if_needed(n_rows, n_cols, rhs);

        for (size_t r = 0; r < n_rows; r++)
        {
            const Node& yaml_row = node[r];
            if (yaml_row.size() != n_cols)
                return false;
            for (size_t c = 0; c < n_cols; c++)
                Converter::access_element(r, c, rhs) = yaml_row[c].as<Scalar>();
        }

        return true;
    }

    static bool decode(const Node& node, EigenType& rhs)
    {
        if (!node.IsSequence())
            return false;

        if (node.size() > 0)
        {
            if (!node[0].IsSequence())
                return decode_1d(node, rhs);
            else
                return decode_2d(node, rhs);
        }
        else
        {
            Converter::resize_if_needed(0, 0, rhs);
            return true;
        }
    }
};

template<>
struct convert<Eigen::Quaterniond>
{
    static Node encode(const Eigen::Quaterniond& rhs)
    {
        return convert<Eigen::Vector4d>::encode(
            Eigen::Vector4d(rhs.w(), rhs.x(), rhs.y(), rhs.z()));
    }

    static bool decode(const Node& node, Eigen::Quaterniond& rhs)
    {
        Eigen::Vector4d v4d;
        if (!convert<Eigen::Vector4d>::decode(node, v4d))
            return false;
        
        rhs.w() = v4d[0];
        rhs.x() = v4d[1];
        rhs.y() = v4d[2];
        rhs.z() = v4d[3];

        return true;
    }
};

}  // namespace YAML