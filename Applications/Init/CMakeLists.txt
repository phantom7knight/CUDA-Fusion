set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20) # Ensure CUDA uses the same standard

find_package(CUDAToolkit)

# Add the executable or target
add_executable(Init main.cu)  # Replace "main.cu" with your CUDA source file

# Link CUDA libraries if needed
target_link_libraries(Init PRIVATE CUDA::cudart)  # cudart is the CUDA runtime library

# Specify properties for CUDA target (optional)
set_target_properties(Init PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	
# Optionally, set compiler options
# For example, enabling optimizations and debugging symbols
# target_compile_options(Init PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_52,code=sm_52>)
# target_compile_options(Init PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)