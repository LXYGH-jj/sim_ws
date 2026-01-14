# CMakeLists.txt

# License: BSD 3-Clause License
# Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

################################################################################
# set up project
################################################################################
cmake_minimum_required(VERSION 3.13)
project(legarm_wbc VERSION 1.0.0)

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