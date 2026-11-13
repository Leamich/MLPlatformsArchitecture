cmake_minimum_required(VERSION 3.20)
project(hw01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for IntelliSense / clangd
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(HW01_BUILD_TESTS "Build correctness tests" ON)
option(HW01_BUILD_BENCH "Build load tests (benchmarks)" ON)

enable_testing()

# ---------------- dependencies ----------------
include(FetchContent)

find_package(hwy REQUIRED)   # Google Highway (apt: libhwy-dev | brew: highway)
find_package(TBB REQUIRED)   # apt: libtbb-dev | brew: tbb

if(HW01_BUILD_TESTS AND NOT TARGET GTest::gtest_main)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

if(HW01_BUILD_BENCH AND NOT TARGET benchmark::benchmark_main)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(benchmark)
endif()

# ---------------- library (header-only) ----------------
add_library(hw01_core INTERFACE)
target_include_directories(hw01_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(hw01_core INTERFACE cxx_std_20)
# -march=native -> Highway "static dispatch": SIMD code is compiled for the best
# instruction set of THIS machine (AVX2 / AVX-512 / NEON ...).
target_compile_options(hw01_core INTERFACE -march=native -Wall -Wextra)
target_link_libraries(hw01_core INTERFACE hwy::hwy TBB::tbb)
# Without this the compiler vectorizes the scalar loops itself, and "scalar vs SIMD" means nothing.
option(HW01_NO_AUTOVEC "Disable compiler auto-vectorization" ON)
if(HW01_NO_AUTOVEC)
  target_compile_options(hw01_core INTERFACE -fno-tree-vectorize -fno-tree-slp-vectorize)
endif()

if(HW01_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(HW01_BUILD_BENCH)
  add_subdirectory(bench)
endif()
