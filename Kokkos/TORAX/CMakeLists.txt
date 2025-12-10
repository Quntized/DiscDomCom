cmake_minimum_required(VERSION 3.14)
project(torax_kokkos LANGUAGES CXX)

# 1. Setup C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- FIX: Explicitly find Python Development Headers ---
# This forces CMake to find the Python.h file associated with your current Interpreter
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
# -------------------------------------------------------

# 2. Find Kokkos
find_package(Kokkos REQUIRED)

# 3. Fetch Nanobind
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.0.0
)
FetchContent_MakeAvailable(nanobind)

# 4. Define the Python Module
nanobind_add_module(torax_kokkos bindings.cpp)

# 5. Link Libraries
target_link_libraries(torax_kokkos PRIVATE Kokkos::kokkos)

# 6. Optimization Flags
if(NOT MSVC)
  target_compile_options(torax_kokkos PRIVATE -O3 -march=native)
endif()
