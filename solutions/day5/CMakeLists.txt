cmake_minimum_required(VERSION 4.0)

project(cuda_app LANGUAGES CUDA CXX)

# -------------------------
# Language standards
# -------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------
# CUDA architecture
# -------------------------
set(CMAKE_CUDA_ARCHITECTURES 75)

# -------------------------
# Compiler flags
# -------------------------

# MSVC host compiler flags
set(HOST_COMPILER_FLAGS
    /W3
    /O2
)

# CUDA flags (passed via NVCC)
set(CUDA_COMPILER_FLAGS
    -Xcompiler=/W3
    -Xcompiler=/O2
)

# -------------------------
# Main executable
# -------------------------
add_executable(main
    src/main.cu
)

target_include_directories(main PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${HOST_COMPILER_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILER_FLAGS}>
)

# -------------------------
# Convenience run targets
# -------------------------
add_custom_target(run_main
    COMMAND $<TARGET_FILE:main>
    DEPENDS main
)