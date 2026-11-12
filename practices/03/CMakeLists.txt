cmake_minimum_required(VERSION 3.10)
project(l3_simd)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(hwy REQUIRED)
find_package(TBB REQUIRED)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.5
)

FetchContent_MakeAvailable(googlebenchmark)

set(SOURCES
    muladd_baseline.cpp
    muladd.cpp
    muladd_benchmark.cpp
    matmul.cpp
    matmul_benchmark.cpp
    activations_benchmark.cpp
)
set_source_files_properties(activations_benchmark.cpp PROPERTIES COMPILE_FLAGS "-march=native")

add_executable(l3_simd ${SOURCES})
target_include_directories(l3_simd PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(l3_simd benchmark::benchmark benchmark::benchmark_main hwy::hwy TBB::tbb)
target_include_directories(l3_simd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(l3_simd PRIVATE
  -O3 -march=native
#   -fopt-info-vec-all # флаг выводит
  -ffast-math
)
