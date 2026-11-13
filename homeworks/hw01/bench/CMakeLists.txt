add_executable(hw01_bench bench.cpp)
target_link_libraries(hw01_bench PRIVATE hw01_core benchmark::benchmark_main)
