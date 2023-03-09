cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(main)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

list (APPEND CMAKE_PREFIX_PATH "/home/tugsa/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
set_property(TARGET main PROPERTY CXX_STANDARD 17)