# The DLSS 5 NR demo: Filament (the patched checkout, installed to third_party/filament-install) + SDL2 + the NR
# host code of src/. Two Vulkan worlds, each its own target with its own headers: `nr` (volk + tools/Vulkan-Headers:
# the device, the NR pass) and `demo_filament_vulkan` (Filament's bundled headers: the shared context, the interop).
# Configured and built by scripts/build_demo.ps1.
cmake_minimum_required(VERSION 3.19)
project(dlss5-demo LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(TOOLS ${ROOT}/tools)
set(FILAMENT ${ROOT}/third_party/filament-install)
set(FILAMENT_SRC ${ROOT}/third_party/filament)
if (NOT EXISTS ${FILAMENT}/include/filament/Engine.h)
    message(FATAL_ERROR "third_party/filament-install is missing: run the fetch_filament and build_filament scripts for your platform")
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH)
string(REPLACE "amd64" "x86_64" ARCH "${ARCH}")
set(FILAMENT_LIB ${FILAMENT}/lib/${ARCH})
if (MSVC)
    # Filament's static libraries are built with /MT
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/EHsc /W3 /wd4244 /wd4267 /utf-8 /Zi)
    add_link_options(/DEBUG)   # a PDB next to the executable: the crash report symbolizes the demo's own frames
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# ---- the NR side: the device, the pass, the kernels' host code (volk, the repository's Vulkan headers)
add_library(nr STATIC
    ${ROOT}/src/vk_context.cpp
    ${ROOT}/src/gpu_arch.cpp
    ${ROOT}/src/kernels.cpp
    ${ROOT}/src/nr_graph.cpp
    ${ROOT}/src/nr_model.cpp
    ${ROOT}/src/reference.cpp
    ${TOOLS}/volk/volk.c
    vulkan_device.cpp
    nr_pass.cpp)
target_include_directories(nr PUBLIC ${TOOLS}/Vulkan-Headers/include ${TOOLS}/volk ${ROOT}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(nr PUBLIC VK_NO_PROTOTYPES VK_ENABLE_BETA_EXTENSIONS)

# ---- the Filament-side Vulkan glue (Filament's bundled headers, through bluevk)
add_library(demo_filament_vulkan STATIC filament_vulkan.cpp)
target_include_directories(demo_filament_vulkan PRIVATE ${FILAMENT}/include ${CMAKE_CURRENT_SOURCE_DIR})

# ---- the application
set(DEMO_SOURCES main.cpp scene.cpp crash_report.cpp)
if (APPLE)
    list(APPEND DEMO_SOURCES native_window_cocoa.mm)
endif()
add_executable(dlss5-demo ${DEMO_SOURCES})
# (the NR headers first: main.cpp reaches Vulkan through volk, never through Filament's copy of the headers)
target_include_directories(dlss5-demo PRIVATE
    ${TOOLS}/Vulkan-Headers/include ${TOOLS}/volk ${ROOT}/src ${CMAKE_CURRENT_SOURCE_DIR}
    ${FILAMENT}/include ${FILAMENT}/include/SDL2 ${FILAMENT}/include/imgui ${FILAMENT_SRC}/third_party/stb)
target_compile_definitions(dlss5-demo PRIVATE VK_NO_PROTOTYPES VK_ENABLE_BETA_EXTENSIONS)
set(FILAMENT_LIBS
    filagui imgui gltfio_core uberarchive uberzlib ktxreader basis_transcoder dracodec meshoptimizer image
    filament-iblprefilter filament backend bluevk smol-v filabridge filaflat geometry mikktspace ibl
    utils abseil zstd stb getopt perfetto sdl2)
foreach(lib ${FILAMENT_LIBS})
    list(APPEND FILAMENT_LIB_FILES ${FILAMENT_LIB}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX})
endforeach()
target_link_libraries(dlss5-demo PRIVATE nr demo_filament_vulkan ${FILAMENT_LIB_FILES})
if (WIN32)
    target_link_libraries(dlss5-demo PRIVATE Shlwapi user32 gdi32 imm32 version winmm setupapi ole32 oleaut32 advapi32 shell32)
    set_target_properties(dlss5-demo PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
elseif (APPLE)
    target_link_libraries(dlss5-demo PRIVATE "-framework Cocoa" "-framework QuartzCore" "-framework Metal" "-framework IOKit"
        "-framework CoreVideo" "-framework Carbon" "-framework AudioToolbox" "-framework ForceFeedback" iconv)
else()
    find_package(Threads REQUIRED)
    find_package(X11 REQUIRED)
    target_link_libraries(dlss5-demo PRIVATE Threads::Threads dl X11::X11)
    foreach(lib Xext Xcursor Xi Xfixes Xrandr Xss)
        find_library(${lib}_LIBRARY NAMES ${lib} REQUIRED)
        target_link_libraries(dlss5-demo PRIVATE ${${lib}_LIBRARY})
    endforeach()
endif()
set_target_properties(dlss5-demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ROOT}/build/demo)
