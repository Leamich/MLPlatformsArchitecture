cmake_minimum_required(VERSION 4.4)
project(eigen-test)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

foreach(src base.cpp perf.cpp)
  get_filename_component(name ${src} NAME_WE)
  add_executable(l2_eigen_${name} ${src})
  target_compile_options(l2_eigen_${name} PRIVATE -O3)
  target_link_libraries(l2_eigen_${name} Eigen3::Eigen)
endforeach()
