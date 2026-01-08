# CMakeLists.txt

# License: BSD 3-Clause License
# Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

################################################################################
# set up the project
################################################################################
cmake_minimum_required(VERSION 3.13)
project(commutils VERSION 1.0.0)

# specify the C++ 17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# display all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
# strongly optimize code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)
# when building, don't use the install RPATH already (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
# add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


################################################################################
# find dependencies
################################################################################
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)


################################################################################
# build and install the main library
################################################################################
# headers only library. we only need to install
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
  INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} 
  INTERFACE
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
)
# command to install the library
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
)
# we also need to install the header files
# here we just copy the whole directory
# we might want to only open a few for other packages (replace DIRECTORY by FILES)
install(DIRECTORY include/ DESTINATION include)


################################################################################
# do unit tests
################################################################################
include(CTest)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)

    add_executable(test_${PROJECT_NAME}
        test/gtest_main.cpp
        test/csv/test_csv_writer.cpp
        test/yaml/test_yaml_tools.cpp
        test/yaml/test_yaml_eigen.cpp
    )
    target_include_directories(test_${PROJECT_NAME}
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(test_${PROJECT_NAME}
      PUBLIC
        Eigen3::Eigen
        ${YAML_CPP_LIBRARIES}
        GTest::gtest
    )
    add_test(TARGET test_${PROJECT_NAME})
endif()


################################################################################
# create the cmake package
################################################################################
include(CMakePackageConfigHelpers)

# generate the necessary cmake file
set(cm_files "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in ${cm_files}
  INSTALL_DESTINATION share/${PROJECT_NAME}/cmake)
# we test if there is a version to be installed
if(DEFINED PROJECT_VERSION)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
  list(APPEND cm_files
       "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
endif()

# we copy the cmake files we would need to configure the project
install(FILES ${cm_files} DESTINATION share/${PROJECT_NAME}/cmake)

# we install the cmake package
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake)


################################################################################
# get the python file installation directory
################################################################################
function(get_python_install_dir output)
  # Find the python interpreter.
  find_package(Python REQUIRED)
  # code to find installation path for python libs
  set(_python_code
      "from distutils.sysconfig import get_python_lib"
      "import os"
      "install_path = '${CMAKE_INSTALL_PREFIX}'"
      "python_lib = get_python_lib(prefix=install_path)"
      "rel_path = os.path.relpath(python_lib, start=install_path)"
      "print(rel_path.replace(os.sep, '/'))")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "-c" "${_python_code}"
    OUTPUT_VARIABLE _output
    RESULT_VARIABLE _result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _result EQUAL 0)
    message(
      FATAL_ERROR
        "execute_process(${Python_EXECUTABLE} -c '${_python_code}') returned "
        "error code ${_result}")
  endif()
  set(${output}
      "${_output}"
      PARENT_SCOPE)
endfunction()

get_python_install_dir(python_install_dir)


################################################################################
# install the python package
################################################################################
# install the python package too
install(
  DIRECTORY python/${PROJECT_NAME}
  DESTINATION "${python_install_dir}"
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE)