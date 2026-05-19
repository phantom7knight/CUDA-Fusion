add_executable(Init
  main.cu
  ../../Helpers/DefaultIncludes.h
  ../../Helpers/helper_cuda.h
  ../../Helpers/helper_functions.h
  ../../Helpers/helper_math.h
  ../../Helpers/helper_string.h
  ../../Helpers/helper_timer.h
  ../../Helpers/driver_types.h
)

set_target_properties(Init PROPERTIES
  CUDA_ARCHITECTURES 75
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_compile_definitions(Init PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:CMAKE_INTDIR=$<CONFIG>>
)

# --- Set VS Debugging properties automatically ---
set_target_properties(Init PROPERTIES
  VS_DEBUGGER_COMMAND "${CMAKE_BINARY_DIR}/Bin/$<CONFIG>/Init.exe"
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)