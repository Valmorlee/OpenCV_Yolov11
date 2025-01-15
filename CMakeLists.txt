cmake_minimum_required(VERSION 3.26)
project(OpenCV_Yolov11)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/local/include/opencv4")

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(OpenCV_Yolov11 main.cpp)

target_link_libraries(OpenCV_Yolov11 ${OpenCV_LIBS})
