# Find ROCmTools libraries/headers

find_path(ROCM_PREFIX
  NAMES include/rocmtools/rocmtools.h
)

find_library(ROCMTOOLS_LIBROCMTOOLS
  NAMES rocmtools
  HINTS ${ROCM_PREFIX}/lib
)
find_library(ROCMTOOLS_LIBHSARUNTIME
  NAMES hsa-runtime64
  HINTS ${ROCM_PREFIX}/lib
)
find_library(ROCMTOOLS_LIBHSAKMT
  NAMES hsakmt
  HINTS ${ROCM_PREFIX}/lib
)

find_path(ROCMTOOLS_INCLUDE_DIR
  NAMES rocmtools.h
  HINTS ${ROCM_PREFIX}/include/rocmtools
)

find_path(HIP_INCLUDE_DIR
  NAMES hip/hip_runtime.h
  HINTS ${ROCM_PREFIX}/include)

set(ROCMTOOLS_INCLUDE_DIRS 
  ${ROCMTOOLS_INCLUDE_DIR}
  ${HIP_INCLUDE_DIR}
)
set(ROCMTOOLS_LIBRARIES
  ${ROCMTOOLS_LIBROCMTOOLS}
  ${ROCMTOOLS_LIBHSARUNTIME}
  ${ROCMTOOLS_LIBHSAKMT})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ROCMTOOLS
  DEFAULT_MSG
  ROCMTOOLS_LIBROCMTOOLS
  ROCMTOOLS_LIBHSARUNTIME
  ROCMTOOLS_LIBHSAKMT
  ROCMTOOLS_INCLUDE_DIRS
)

mark_as_advanced(
  ROCMTOOLS_INCLUDE_DIRS
  ROCMTOOLS_LIBRARIES
)
