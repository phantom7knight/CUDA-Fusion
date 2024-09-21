cmake_minimum_required(VERSION 3.18 FATAL_ERROR)  # CMake 3.18+ recommended for CUDA support
project(CUDA_Fusion LANGUAGES CXX CUDA)

# Enable CUDA
find_package(CUDA REQUIRED)

# Set the CUDA architecture (Optional, can be automatic in modern CMake)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 80)  # Change based on your target GPU

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# Specify the C++ Standard
set(CMAKE_CXX_STANDARD 20)  # Or any other version like 17, 20
set(CMAKE_CUDA_STANDARD 20) # Ensure CUDA uses the same standard

# Specify output paths for all configurations (same for CUDA projects)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin)    # For static libraries (.lib, .a)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin)    # For shared libraries (.so, .dll)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin)    # For executables (.exe, .out)

add_subdirectory(Applications)
