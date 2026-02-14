cmake_minimum_required(VERSION 3.22)
project(ddc_educational_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────────────────────────────
# Find DDC (which transitively brings in Kokkos)
# ──────────────────────────────────────────────────────────────────────
#
# If DDC is installed system-wide or via CMAKE_PREFIX_PATH:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/ddc/install ..
#
# If using DDC as a subdirectory (e.g. git submodule):
#   Uncomment the add_subdirectory line below instead.
# ──────────────────────────────────────────────────────────────────────

# Option A: DDC installed externally (recommended)
find_package(DDC REQUIRED)

# Option B: DDC as a subdirectory (uncomment if using git submodule)
# add_subdirectory(vendor/ddc)

# ──────────────────────────────────────────────────────────────────────
# Build the example
# ──────────────────────────────────────────────────────────────────────
add_executable(sajid main.cpp)

target_link_libraries(sajid PRIVATE DDC::DDC)

# Kokkos requires this for CUDA builds
# (no-op for CPU-only builds)
if(Kokkos_ENABLE_CUDA)
    target_compile_options(sajid PRIVATE --expt-relaxed-constexpr)
endif()
