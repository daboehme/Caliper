set(CALIPER_MONITOR_SOURCES
  LoopMonitor.cpp
  RegionMonitor.cpp
  Timeseries.cpp)

add_caliper_service("loop_monitor")
add_caliper_service("region_monitor")
add_caliper_service("timeseries")

if (CALIPER_HAVE_PTHREADS)
  list(APPEND CALIPER_MONITOR_SOURCES
    ThreadMonitor.cpp)
  add_caliper_service("thread_monitor")
endif()

add_service_sources(${CALIPER_MONITOR_SOURCES})