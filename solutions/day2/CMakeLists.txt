cmake_minimum_required(VERSION 4.0)

project(cuda_app LANGUAGES CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

# Set all host compiler flags in one variable
set(CUDA_HOST_COMPILER_FLAGS 
    -Xcompiler=/W3 # Compiler Warnings
    -Xcompiler=/O2 # Performance Optimization
)

add_executable(cuda_app
    src/main.cu
    src/kernels.cu
)

target_include_directories(cuda_app PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(cuda_app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_COMPILER_FLAGS}>
)

add_custom_target(run
    COMMAND $<TARGET_FILE:cuda_app>
    DEPENDS cuda_app
)