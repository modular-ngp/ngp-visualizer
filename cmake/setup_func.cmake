function(add_vv_example)
    set(options)
    set(oneValueArgs NAME SRC SHADER_BIN_DIR)
    set(multiValueArgs SHADERS)
    cmake_parse_arguments(AV "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT AV_NAME OR NOT AV_SRC)
        message(FATAL_ERROR "add_vv_example requires NAME and SRC")
    endif ()

    add_executable(${AV_NAME} ${AV_SRC})
    target_link_libraries(${AV_NAME} PRIVATE VulkanVisualizer::vulkan_visualizer)

    if (AV_SHADERS)
        if (AV_SHADER_BIN_DIR)
            set(_BIN_DIR "${AV_SHADER_BIN_DIR}")
        else ()
            set(_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}")
        endif ()
        find_program(GLSLC glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)
        if (NOT GLSLC)
            message(FATAL_ERROR "glslc not found")
        endif ()
        set(SPV_FILES)
        foreach (SH ${AV_SHADERS})
            string(REPLACE "\\" "/" SH_NORM "${SH}")
            set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${SH_NORM}")
            set(SPV_FILE "${_BIN_DIR}/${SH_NORM}.spv")
            get_filename_component(SPV_DIR "${SPV_FILE}" DIRECTORY)
            file(MAKE_DIRECTORY "${SPV_DIR}")
            add_custom_command(OUTPUT ${SPV_FILE}
                    COMMAND ${GLSLC} -O -c ${SRC_FILE} -o ${SPV_FILE}
                    DEPENDS ${SRC_FILE}
                    VERBATIM)
            list(APPEND SPV_FILES ${SPV_FILE})
        endforeach ()
        if (SPV_FILES)
            add_custom_target(${AV_NAME}_shaders DEPENDS ${SPV_FILES})
            add_dependencies(${AV_NAME} ${AV_NAME}_shaders)
        endif ()
    endif ()

    if (WIN32)
        add_custom_command(TARGET ${AV_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${AV_NAME}> $<TARGET_FILE_DIR:${AV_NAME}>
                COMMAND_EXPAND_LISTS)
    endif ()
endfunction()
