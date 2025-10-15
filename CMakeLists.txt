cmake_minimum_required(VERSION 3.26)
include(cmake/bootstrap.cmake)
project(ngp-visualizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/vulkan-visualizer-sdk")
find_package(VulkanVisualizer CONFIG REQUIRED)

include(cmake/setup_func.cmake)
add_vv_example(
        NAME ngp-visualizer
        SRC main.cpp
        SHADERS shaders/point_cloud.vert shaders/point_cloud.frag
)

include(FetchContent)
FetchContent_Declare(
        dataset
        GIT_REPOSITORY https://github.com/modular-ngp/dataset.git
        GIT_TAG master
)
FetchContent_MakeAvailable(dataset)
target_link_libraries(ngp-visualizer PRIVATE dataset::dataset)
