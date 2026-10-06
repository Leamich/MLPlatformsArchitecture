cmake_minimum_required(VERSION 3.10)
project(hw01 LANGUAGES CXX)

find_package(Threads REQUIRED)

## CPU PROBE

add_executable(cpu_probe src/cpu_probe.cpp)
target_link_libraries(cpu_probe PRIVATE Threads::Threads)

## OpenCL Check

find_package(OpenCL REQUIRED)

add_executable(cl_query src/main_cl.cpp)
target_compile_definitions(cl_query PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_link_libraries(cl_query PRIVATE OpenCL::OpenCL)