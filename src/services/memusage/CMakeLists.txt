set(CALIPER_MEMUSAGE_SOURCES
  MemStatService.cpp
  MemUsageService.cpp)

add_service_sources(${CALIPER_MEMUSAGE_SOURCES})
add_caliper_service("memusage")
add_caliper_service("memstat")
