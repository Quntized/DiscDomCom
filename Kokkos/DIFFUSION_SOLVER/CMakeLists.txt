cmake_minimum_required(VERSION 3.20)
project(KokkosDiffusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. Find Dependencies
find_package(Kokkos REQUIRED)
find_package(pybind11 REQUIRED)

# 2. Build the Python Module (.so file)
# This one was already working (mostly)
pybind11_add_module(diffusion_kokkos diffusion_kokkos_py.cpp)
target_link_libraries(diffusion_kokkos PRIVATE Kokkos::kokkos)

# 3. Build the CLI Executable (The part that failed)
add_executable(diffusion_kokkos_cli diffusion_kokkos_cli.cpp)

# CRITICAL FIX: Link against pybind11::embed (or pybind11::lto) 
# AND Kokkos. 'embed' provides Python.h and libpython.
target_link_libraries(diffusion_kokkos_cli PRIVATE 
    Kokkos::kokkos
    pybind11::embed
)

# Optional: Add strict warning flags if you like, but the above fixes the error.
