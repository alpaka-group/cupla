################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.11.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.
################################################################################

project(separate_compilation)

################################################################################
# Find cupla
################################################################################

list(APPEND CMAKE_MODULE_PATH "${cupla_ROOT}")
find_package("cupla" REQUIRED)


################################################################################
# Add library.
################################################################################

cupla_add_library(${PROJECT_NAME} STATIC "src/kernel.cpp")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
