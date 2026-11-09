cmake_minimum_required(VERSION 3.30)
project(Spire LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# JSON parsing for .safetensors headers. Prefer a system package; fall back to
# a pinned FetchContent download.
find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.12.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Google Benchmark for performance measurements.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.10.0
    )
    # Benchmark depends on GTest for some tests; we disable them to avoid nested dependencies.
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Blackwell GPUs by default: 100 = B200, 120 = RTX 50 series (RTX 5060, 5090).
# Override from the command line for faster local builds, e.g. -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "100;120")
endif()

add_library(spire_core
    src/daemon.cpp
    src/kernels/sae_encoder.cu
    src/kernels/dtype_convert.cu
    src/SafetensorsFile.cpp
    src/mapped_file.cpp
    src/load_safetensors.cpp
)
set_target_properties(spire_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(spire_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(spire_core PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)

add_executable(spire src/main.cpp)
target_link_libraries(spire PRIVATE spire_core CUDA::cudart)

option(BUILD_PYTHON_BINDINGS "Build the Python bindings module" OFF)

if(BUILD_PYTHON_BINDINGS)
    # Determine which Python interpreter to use for building bindings.
    # Priority: 1. User-provided SPIRE_PYTHON_EXECUTABLE, 2. Current active python3, 3. Fallback search.
    if(NOT SPIRE_PYTHON_EXECUTABLE)
        find_program(SPIRE_PYTHON_EXECUTABLE 
                     NAMES python3 python 
                     ENV PYTHON_EXECUTABLE)
    endif()

    if(NOT SPIRE_PYTHON_EXECUTABLE)
        message(FATAL_ERROR "BUILD_PYTHON_BINDINGS=ON but no Python interpreter was found.")
    endif()
    
    message(STATUS "Using Python interpreter: ${SPIRE_PYTHON_EXECUTABLE}")

    # Pybind11 for Python bindings.
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    # Resolve Torch paths from the selected Python environment.
    execute_process(
        COMMAND "${SPIRE_PYTHON_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE torch_res
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(torch_res EQUAL 0)
        execute_process(
            COMMAND "${SPIRE_PYTHON_EXECUTABLE}" -c "import torch; print(torch.__path__[0])"
            OUTPUT_VARIABLE TORCH_PYTHON_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND "${SPIRE_PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
            OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(TORCH_INCLUDE_DIR
            "${TORCH_PYTHON_ROOT}/include"
            "${TORCH_PYTHON_ROOT}/include/torch/csrc/api/include")
        set(TORCH_PYTHON_LIBRARY "${TORCH_PYTHON_ROOT}/lib/libtorch_python.so")
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
        find_package(Torch CONFIG REQUIRED)
    else()
        message(FATAL_ERROR
            "BUILD_PYTHON_BINDINGS=ON but PyTorch was not found using "
            "${SPIRE_PYTHON_EXECUTABLE}.")
    endif()

    # Python bindings module
    add_library(spire_cpp MODULE src/bindings.cpp)
    target_link_libraries(spire_cpp PRIVATE
        spire_core CUDA::cudart pybind11::pybind11 "${TORCH_LIBRARIES}"
        "${TORCH_PYTHON_LIBRARY}")
    target_include_directories(spire_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIR})
    set_target_properties(spire_cpp PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        BUILD_RPATH "${TORCH_PYTHON_ROOT}/lib")
endif()

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)