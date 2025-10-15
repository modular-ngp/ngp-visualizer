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

add_library(shmx INTERFACE)
add_library(shmx::shmx ALIAS shmx)

find_path(SHMX_INCLUDE_DIR
        NAMES shmx_client.h
        PATHS
            ${CMAKE_CURRENT_BINARY_DIR}/_deps/shared-mem-ipc-src/src
            ${CMAKE_SOURCE_DIR}/cmake-build-release/_deps/shared-mem-ipc-src/src
            ${CMAKE_SOURCE_DIR}/_deps/shared-mem-ipc-src/src
        NO_DEFAULT_PATH)

if (NOT SHMX_INCLUDE_DIR)
    message(FATAL_ERROR "shmx_client.h not found. Ensure shared-mem-ipc has been prepared under _deps.")
endif ()

target_include_directories(shmx INTERFACE ${SHMX_INCLUDE_DIR})
target_link_libraries(ngp-visualizer PRIVATE shmx::shmx)
