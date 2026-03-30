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
# Vector Addition executable
# -------------------------
add_executable(vector_add
    src/vector_add.cu
)

target_include_directories(vector_add PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${HOST_COMPILER_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILER_FLAGS}>
)

# -------------------------
# Matrix Multiplication executable
# -------------------------
add_executable(matrix_mult
    src/matrix_mult.cu
)

target_include_directories(matrix_mult PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(matrix_mult PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${HOST_COMPILER_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILER_FLAGS}>
)

# -------------------------
# Convenience run targets
# -------------------------
add_custom_target(run_vector_add
    COMMAND $<TARGET_FILE:vector_add>
    DEPENDS vector_add
)

add_custom_target(run_matrix_mult
    COMMAND $<TARGET_FILE:matrix_mult>
    DEPENDS matrix_mult
)