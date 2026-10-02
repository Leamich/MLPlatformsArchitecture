cmake_minimum_required(VERSION 4.4)
project(torch-example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(Torch REQUIRED)
message(STATUS "Torch found: ${Torch_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(l2_torch torch.cpp)
target_include_directories(l2_torch PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(l2_torch ${TORCH_LIBRARIES})

add_executable(l2_torch_tensors tensors.cpp)
target_include_directories(l2_torch_tensors PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(l2_torch_tensors ${TORCH_LIBRARIES})

add_executable(l2_torch_dataloader dataloader.cpp)
target_include_directories(l2_torch_dataloader PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(l2_torch_dataloader ${TORCH_LIBRARIES})
