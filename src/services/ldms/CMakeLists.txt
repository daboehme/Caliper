set(CALIPER_LDMS_SOURCES
  LdmsForwarder.cpp)

add_service_sources(${CALIPER_LDMS_SOURCES})
add_caliper_service("ldms")
