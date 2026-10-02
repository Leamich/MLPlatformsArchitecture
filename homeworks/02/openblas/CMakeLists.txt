cmake_minimum_required(VERSION 4.4)
project(openblas-test)

set(BLA_VENDOR OpenBLAS)
#find_package(OpenBLAS REQUIRED)
find_package(BLAS REQUIRED)

foreach(src sgemm.cpp naive_vs_blas.cpp)
  get_filename_component(name ${src} NAME_WE)
  add_executable(l2_openblas_${name} ${src})
  target_compile_options(l2_openblas_${name} PRIVATE -O3)
  target_link_libraries(l2_openblas_${name} PRIVATE BLAS::BLAS)
  # target_link_libraries(l2_openblas_${name} PRIVATE OpenBLAS::OpenBLAS)
endforeach()