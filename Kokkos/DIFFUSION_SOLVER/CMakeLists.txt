cmake_minimum_required(VERSION 3.15)
project(kokkos_diffusion_py LANGUAGES CXX)

# Require pybind11 and Kokkos
find_package(pybind11 REQUIRED)
find_package(Kokkos REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build Python extension module using pybind11
pybind11_add_module(diffusion_kokkos MODULE diffusion_kokkos_py.cpp)
# Link Kokkos into the module
target_link_libraries(diffusion_kokkos PRIVATE Kokkos::kokkos)
# Ensure PIC and position-independent code for module
set_target_properties(diffusion_kokkos PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Optional CLI executable for running from terminal
add_executable(diffusion_kokkos_cli diffusion_kokkos_cli.cpp)
target_link_libraries(diffusion_kokkos_cli PRIVATE Kokkos::kokkos)
set_target_properties(diffusion_kokkos_cli PROPERTIES CXX_EXTENSIONS OFF)

# Helpful compile flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(diffusion_kokkos_cli PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(diffusion_kokkos PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation hints
include(GNUInstallDirs)
install(TARGETS diffusion_kokkos_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# Note: Python module installation is left to the user/environment (pip, cmake install, etc.)
