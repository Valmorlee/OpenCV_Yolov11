cmake_minimum_required(VERSION 3.26)
project(OpenCV_Yolov11)

# EXPLANATION
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# CUDA
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CXX_STANDARD 20)

# CUDA & OpenCV Package
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/include/opencv4")
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(OpenCV_Yolov11 main.cpp)

# LIB
target_link_libraries(OpenCV_Yolov11 ${OpenCV_LIBS})
