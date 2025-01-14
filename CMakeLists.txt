cmake_minimum_required(VERSION 3.30)
project(OpenCV_Yolov11)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/opt/opencv4.10.0/include/opencv4")

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(OpenCV_Yolov11 main.cpp)

target_link_libraries(OpenCV_Yolov11 ${OpenCV_LIBS})
