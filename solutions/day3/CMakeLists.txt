cmake_minimum_required(VERSION 4.0)

project(cuda_app LANGUAGES C CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

# -------------------------
# CPU executable
# -------------------------
add_executable(cpu_app
    src/main_cpu.c
)

target_include_directories(cpu_app PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# flags for MSVC C compiler
target_compile_options(cpu_app PRIVATE
    /W3 
    /O2
)

target_link_options(cpu_app PRIVATE
    /LTCG
)

# -------------------------
# GPU executable
# -------------------------
add_executable(gpu_app
    src/main_gpu.cu
)

target_include_directories(gpu_app PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Host compiler flags for CUDA (MSVC)
set(CUDA_HOST_COMPILER_FLAGS 
    -Xcompiler=/W3
    -Xcompiler=/O2
)

target_compile_options(gpu_app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_COMPILER_FLAGS}>
)

# -------------------------
# Convenience run targets
# -------------------------
add_custom_target(run_cpu
    COMMAND $<TARGET_FILE:cpu_app>
    DEPENDS cpu_app
)

add_custom_target(run_gpu
    COMMAND $<TARGET_FILE:gpu_app>
    DEPENDS gpu_app
)