set(SOURCES
  CUDATrace.cu
  CaliperCUDATrace.cu)

set(CMAKE_CUDA_STANDARD 11)

add_library(caliper-cudatrace ${SOURCES})

set_target_properties(caliper-cudatrace PROPERTIES 
  CUDA_ARCHITECTURES "70"
  CUDA_SEPARABLE_COMPILATION ON)

add_service_objlib("caliper-cudatrace")
add_caliper_service("cudatrace CALIPER_HAVE_CUDATRACE")
