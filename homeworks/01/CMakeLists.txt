cmake_minimum_required(VERSION 3.10)
project(cpu_probe LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(cpu_probe cpu_probe.cpp)
target_link_libraries(cpu_probe PRIVATE Threads::Threads)
