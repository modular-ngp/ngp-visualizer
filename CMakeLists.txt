cmake_minimum_required(VERSION 3.26)
include(cmake/bootstrap.cmake)
project(vulkan-visualizer-template VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/vulkan-visualizer-sdk")
find_package(VulkanVisualizer CONFIG REQUIRED)

include(cmake/setup_func.cmake)
add_vv_example(
        NAME example_template
        SRC main.cpp
        SHADERS shaders/triangle.vert shaders/triangle.frag
)
