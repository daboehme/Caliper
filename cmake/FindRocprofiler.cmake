# Find ROCProfiler libraries/headers

find_path(ROCM_ROOT_DIR
  NAMES include/rocprofiler/v2/rocprofiler.h
)

find_library(ROCPROFILER_LIBROCPROFILER
  NAMES librocprofiler64.so.2
  HINTS ${ROCM_ROOT_DIR}/lib
)
find_library(ROCPROFILER_LIBHSARUNTIME
  NAMES hsa-runtime64
  HINTS ${ROCM_ROOT_DIR}/lib
)
find_library(ROCPROFILER_LIBHSAKMT
  NAMES hsakmt
  HINTS ${ROCM_ROOT_DIR}/lib
)

find_path(ROCPROFILER_INCLUDE_DIR
  NAMES rocprofiler.h
  HINTS ${ROCM_ROOT_DIR}/include/rocprofiler/v2
)

find_path(HIP_INCLUDE_DIR
  NAMES hip/hip_runtime.h
  HINTS ${ROCM_ROOT_DIR}/include)

set(ROCPROFILER_INCLUDE_DIRS 
  ${ROCPROFILER_INCLUDE_DIR}
  ${HIP_INCLUDE_DIR}
)
set(ROCPROFILER_LIBRARIES
  ${ROCPROFILER_LIBROCPROFILER})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ROCPROFILER
  DEFAULT_MSG
  ROCPROFILER_LIBROCPROFILER
  ROCPROFILER_INCLUDE_DIRS
)

mark_as_advanced(
  ROCPROFILER_INCLUDE_DIRS
  ROCPROFILER_LIBRARIES
)
