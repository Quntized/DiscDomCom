cmake_minimum_required(VERSION 3.15)
project(TDTransportDDC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Optionally use DDC as a subdirectory if provided by user (recommended)
if(EXISTS "${CMAKE_SOURCE_DIR}/ddc/CMakeLists.txt")
  add_subdirectory(${CMAKE_SOURCE_DIR}/ddc ddc_build EXCLUDE_FROM_ALL)
  set(DDC_AVAILABLE TRUE)
endif()

add_library(td_ddc_core
  src/TimeDerivative.cpp
  src/StreamingOperator.cpp
  src/TotalXSOperator.cpp
  src/ScatteringOperator.cpp
  src/FissionOperator.cpp
  src/ExternalSource.cpp
  src/TransportEquation.cpp
)

target_include_directories(td_ddc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(td_ddc_core PUBLIC Kokkos::kokkos)

if(TARGET ddc)
  target_link_libraries(td_ddc_core PUBLIC ddc)
endif()

pybind11_add_module(tdtransport_ddc src/python_bindings.cpp)
target_link_libraries(tdtransport_ddc PRIVATE td_ddc_core)
target_include_directories(tdtransport_ddc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(tdtransport_ddc_native examples/native_test.cpp)
target_link_libraries(tdtransport_ddc_native PRIVATE td_ddc_core)
target_include_directories(tdtransport_ddc_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
