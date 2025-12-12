cmake_minimum_required(VERSION 3.15)
project(TDTransportKokkos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

pybind11_add_module(tdtransport
  src/python_bindings.cpp
  src/TransportSolver.cpp
)

target_include_directories(tdtransport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(tdtransport PRIVATE Kokkos::kokkos)

# Small native executable for quick native tests (optional)
add_executable(tdtransport_native src/native_main.cpp)
target_link_libraries(tdtransport_native PRIVATE Kokkos::kokkos)
target_include_directories(tdtransport_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
