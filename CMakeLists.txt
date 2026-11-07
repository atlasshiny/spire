cmake_minimum_required(VERSION 3.30)
project(Spire LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Blackwell GPUs by default: 100 = B200, 120 = RTX 50 series (RTX 5060, 5090).
# Override from the command line for faster local builds, e.g. -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "100;120")
endif()

# Define library target
add_library(spire_core
    src/daemon.cpp
    src/kernels/sae_encoder.cu
)

# Set public include paths
target_include_directories(spire_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link CUDA runtime library
target_link_libraries(spire_core PRIVATE CUDA::cudart)

# Define executable target
add_executable(spire src/main.cpp)

target_link_libraries(spire PRIVATE spire_core CUDA::cudart)