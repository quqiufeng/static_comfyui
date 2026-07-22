cmake_minimum_required(VERSION 3.18)
project(sd_backend CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Paths to /opt/sd (stable-diffusion.cpp)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dynamic vs static backend mode
# - SD_BACKEND_DL=ON:  use /opt/sd/build-dl with shared libs + dynamic backend plugins
#                      (libggml-cuda.so loaded at runtime, smaller CPU-only package)
# - SD_BACKEND_DL=OFF: use /opt/sd/build with static libs (legacy, all-in-one)
# ---------------------------------------------------------------------------
set(SD_DIR "/opt/sd" CACHE PATH "Path to stable-diffusion.cpp source")
option(SD_BACKEND_DL "Use shared sd.cpp/ggml with dynamic backend loading" ON)

if(SD_BACKEND_DL)
    set(SD_BUILD_DIR_DEFAULT "${SD_DIR}/build-dl")
else()
    set(SD_BUILD_DIR_DEFAULT "${SD_DIR}/build")
endif()
set(SD_BUILD_DIR "${SD_BUILD_DIR_DEFAULT}" CACHE PATH "Path to stable-diffusion.cpp build directory")

message(STATUS "SD_BACKEND_DL: ${SD_BACKEND_DL}")
message(STATUS "SD_BUILD_DIR:  ${SD_BUILD_DIR}")

if(NOT EXISTS ${SD_DIR})
    message(FATAL_ERROR "SD_DIR not found: ${SD_DIR}")
endif()

if(NOT EXISTS ${SD_BUILD_DIR})
    message(FATAL_ERROR "SD_BUILD_DIR not found: ${SD_BUILD_DIR}. Please run build_sd.sh or build_sd_dl.sh first.")
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
set(CUDA_LIB_DIR "${CUDAToolkit_INCLUDE_DIRS}/../lib")
find_library(CUDA_DRIVER_LIB cuda PATHS "${CUDA_LIB_DIR}/stubs" NO_DEFAULT_PATH REQUIRED)
find_library(CULIBOS_LIB culibos PATHS ${CUDA_LIB_DIR} NO_DEFAULT_PATH REQUIRED)

# ---------------------------------------------------------------------------
# Libraries from /opt/sd
# ---------------------------------------------------------------------------
if(SD_BACKEND_DL)
    # Shared libs: stable-diffusion, ggml, ggml-base; backends are loaded at runtime
    find_library(SD_LIB stable-diffusion PATHS ${SD_BUILD_DIR}/bin NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIB ggml PATHS ${SD_BUILD_DIR}/bin NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIB ggml-base PATHS ${SD_BUILD_DIR}/bin NO_DEFAULT_PATH REQUIRED)
    set(GGML_CPU_LIB "")
    set(GGML_CUDA_LIB "")
else()
    # Static libs: everything linked into libsdcpp_adapter.so
    find_library(SD_LIB stable-diffusion PATHS ${SD_BUILD_DIR} NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIB ggml PATHS ${SD_BUILD_DIR}/ggml/src NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CPU_LIB ggml-cpu PATHS ${SD_BUILD_DIR}/ggml/src NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_BASE_LIB ggml-base PATHS ${SD_BUILD_DIR}/ggml/src NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_CUDA_LIB ggml-cuda PATHS ${SD_BUILD_DIR}/ggml/src/ggml-cuda NO_DEFAULT_PATH REQUIRED)
endif()

message(STATUS "SD_LIB:        ${SD_LIB}")
message(STATUS "GGML_LIB:      ${GGML_LIB}")
message(STATUS "GGML_CPU_LIB:  ${GGML_CPU_LIB}")
message(STATUS "GGML_BASE_LIB: ${GGML_BASE_LIB}")
message(STATUS "GGML_CUDA_LIB: ${GGML_CUDA_LIB}")
message(STATUS "CUDA_DRIVER:   ${CUDA_DRIVER_LIB}")
message(STATUS "CULIBOS:       ${CULIBOS_LIB}")

# ---------------------------------------------------------------------------
# Post-processing library (clarity / sharpen / smart-sharpen / edge-sharpen)
# ---------------------------------------------------------------------------
add_library(postproc STATIC
    src/postproc/postproc.cpp
)

target_include_directories(postproc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/postproc
)

# ---------------------------------------------------------------------------
# Adapter library: shared build for StaticPy FFI + static build for C++ examples
# ---------------------------------------------------------------------------
add_library(sdcpp_adapter STATIC
    src/adapters/sdcpp_adapter.cpp
)

add_library(sdcpp_adapter_shared SHARED
    src/adapters/sdcpp_adapter.cpp
)
set_target_properties(sdcpp_adapter_shared PROPERTIES OUTPUT_NAME "sdcpp_adapter")

foreach(target sdcpp_adapter sdcpp_adapter_shared)
    target_include_directories(${target}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/adapters
            ${CMAKE_CURRENT_SOURCE_DIR}/src/postproc
            ${SD_DIR}/include
            ${SD_DIR}
            ${SD_DIR}/thirdparty
    )
endforeach()

# The shared adapter links to sd.cpp / ggml shared libs; CUDA backend is loaded at runtime.
if(SD_BACKEND_DL)
    target_link_libraries(sdcpp_adapter_shared PRIVATE
        postproc
        ${SD_LIB}
        webp
        webpmux
        ${GGML_LIB}
        ${GGML_BASE_LIB}
        dl
        gomp
        pthread
        webp
        sharpyuv
        png
        m
    )
else()
    target_link_libraries(sdcpp_adapter_shared PRIVATE
        ${SD_LIB}
        webp
        webpmux
        ${GGML_LIB}
        dl
        ${GGML_CPU_LIB}
        ${GGML_CUDA_LIB}
        ${GGML_BASE_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        ${CULIBOS_LIB}
        ${CUDA_DRIVER_LIB}
        gomp
        pthread
        webp
        sharpyuv
        png
        m
    )
endif()

# ---------------------------------------------------------------------------
# Common link libraries for sd.cpp executables
# Link order derived from /opt/sd/build/examples/cli link.txt (sd-cli):
#   stable-diffusion -> image libs (webp, webpmux, ...) -> ggml -> dl ->
#   ggml-cpu -> ggml-cuda -> ggml-base -> gomp/pthread ->
#   cudart/cublas/cublasLt/culibos/cuda -> webp/sharpyuv -> m
# ---------------------------------------------------------------------------
# Common link libraries for sd.cpp executables
if(SD_BACKEND_DL)
    set(SD_COMMON_LINK_LIBS
        sdcpp_adapter
        postproc
        ${SD_LIB}
        webp
        webpmux
        ${GGML_LIB}
        ${GGML_BASE_LIB}
        dl
        gomp
        pthread
        webp
        sharpyuv
        png
        m
    )
else()
    set(SD_COMMON_LINK_LIBS
        sdcpp_adapter
        postproc
        ${SD_LIB}
        webp
        webpmux
        ${GGML_LIB}
        dl
        ${GGML_CPU_LIB}
        ${GGML_CUDA_LIB}
        ${GGML_BASE_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        ${CULIBOS_LIB}
        ${CUDA_DRIVER_LIB}
        gomp
        pthread
        webp
        sharpyuv
        png
        m
    )
endif()

function(add_sd_example target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE ${SD_COMMON_LINK_LIBS})
endfunction()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
add_sd_example(sdxl_txt2img examples/sdxl_txt2img.cpp)
add_sd_example(img_hires examples/img_hires.cpp)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS sdcpp_adapter sdcpp_adapter_shared sdxl_txt2img img_hires
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
