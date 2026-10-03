cmake_minimum_required(VERSION 3.24)
project(dlss5vk LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(LOCAL_VULKAN_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/tools/Vulkan-Headers")
set(LOCAL_VOLK "${CMAKE_CURRENT_SOURCE_DIR}/tools/volk")

if(EXISTS "${LOCAL_VULKAN_HEADERS}/include/vulkan/vulkan.h" AND EXISTS "${LOCAL_VOLK}/volk.c")
    set(VULKAN_HEADERS_DIR "${LOCAL_VULKAN_HEADERS}/include")
    set(VOLK_DIR "${LOCAL_VOLK}")
else()
    FetchContent_Declare(
        vulkan_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG v1.4.363
        GIT_SHALLOW TRUE)
    FetchContent_Declare(
        volk_source
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG vulkan-sdk-1.4.357.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(vulkan_headers volk_source)
    set(VULKAN_HEADERS_DIR "${vulkan_headers_SOURCE_DIR}/include")
    set(VOLK_DIR "${volk_source_SOURCE_DIR}")
endif()

add_library(dlss5vk_volk STATIC "${VOLK_DIR}/volk.c")
target_include_directories(dlss5vk_volk PUBLIC "${VULKAN_HEADERS_DIR}" "${VOLK_DIR}")
target_compile_definitions(dlss5vk_volk PUBLIC VK_NO_PROTOTYPES VK_ENABLE_BETA_EXTENSIONS)
if(UNIX AND NOT APPLE)
    target_link_libraries(dlss5vk_volk PUBLIC dl)
endif()

# The Direct3D 12 / 11 adapters (NVIDIA's CUDA interop through the driver's nvapi64.dll, resolved at run time).
set(DLSS5VK_D3D_SOURCES "")
if(WIN32)
    set(DLSS5VK_D3D_SOURCES src/d3d/d3d_common.cpp src/d3d/d3d12_device.cpp src/d3d/d3d11_device.cpp)
endif()

add_executable(dlss5vk
    ${DLSS5VK_D3D_SOURCES}
    src/device_factory.cpp
    src/exec.cpp
    src/gpu_arch.cpp
    src/kernels.cpp
    src/main.cpp
    src/nr_graph.cpp
    src/nr_model.cpp
    src/reference.cpp
    src/verify.cpp
    src/vk_context.cpp)
target_include_directories(dlss5vk PRIVATE src)
target_link_libraries(dlss5vk PRIVATE dlss5vk_volk)
if(WIN32)
    target_link_libraries(dlss5vk PRIVATE d3d12 d3d11 dxgi d3dcompiler dxguid)
endif()
target_compile_definitions(dlss5vk PRIVATE VK_NO_PROTOTYPES VK_ENABLE_BETA_EXTENSIONS)

if(MSVC)
    target_compile_options(dlss5vk PRIVATE /EHsc /W3)
    target_compile_definitions(dlss5vk PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(dlss5vk PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

set_target_properties(dlss5vk PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
