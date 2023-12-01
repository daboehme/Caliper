add_library(caliper-rocprofiler OBJECT Rocprofiler.cpp)
target_compile_definitions(caliper-rocprofiler PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(caliper-rocprofiler PRIVATE ${ROCPROFILER_INCLUDE_DIRS})

add_service_objlib("caliper-rocprofiler")
add_caliper_service("rocprofiler CALIPER_HAVE_ROCPROFILER")
