cmake_minimum_required(VERSION 3.14)
project(UltraFastFeaturization)

# Find required packages
find_package(pybind11 REQUIRED)

#find_package(HDF5 COMPONENTS C CXX REQUIRED)
# Set HDF5 custom include and library directories
#include_directories("/home/ajinkya.hire/ls/envs/uf3_env/include")
#link_directories("/home/ajinkya.hire/ls/envs/uf3_env/lib")
# Require the user to specify custom HDF5 directories
set(HDF5_INCLUDE_DIR "" CACHE PATH "Path to HDF5 include directory")
set(HDF5_LIB_DIR "" CACHE PATH "Path to HDF5 library directory")

# Check if HDF5 directories are set
if(NOT HDF5_INCLUDE_DIR)
    message(FATAL_ERROR "HDF5 include directory not specified. Please set HDF5_INCLUDE_DIR.")
endif()

if(NOT HDF5_LIB_DIR)
    message(FATAL_ERROR "HDF5 library directory not specified. Please set HDF5_LIB_DIR.")
endif()

# Use the specified HDF5 directories
include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIB_DIR})


# Define the output directory for the shared library
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/uf3)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source directory
include_directories(${PROJECT_SOURCE_DIR}/src)

# file(GLOB...) for wildcard additions:
file(GLOB SOURCES "../src/*.cpp")

# Define the target library
pybind11_add_module(ultra_fast_featurize ${SOURCES})

# Link against HDF5 libraries
target_link_libraries(ultra_fast_featurize PRIVATE hdf5_cpp)
