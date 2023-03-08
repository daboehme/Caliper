set(CALIPER_ROCMTOOLS_SOURCES
  Rocm.cpp)

add_library(caliper-rocmtools OBJECT ${CALIPER_ROCMTOOLS_SOURCES})
target_compile_definitions(caliper-rocmtools PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(caliper-rocmtools PRIVATE ${ROCMTOOLS_INCLUDE_DIRS})

add_service_objlib("caliper-rocmtools")
add_caliper_service("rocmtools CALIPER_HAVE_ROCMTOOLS")
