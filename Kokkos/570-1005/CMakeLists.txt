cmake_minimum_required(VERSION 3.15)
project(kokkos_view_examples_full LANGUAGES CXX)

# Find Kokkos (user should pass -DKokkos_DIR if necessary)
find_package(Kokkos REQUIRED)

add_executable(view_examples_full view_examples_full.cpp)
set_target_properties(view_examples_full PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(view_examples_full PRIVATE cxx_std_17)

# Link Kokkos
target_link_libraries(view_examples_full Kokkos::kokkos)

# For nicer output when running under MPI/remote builds
if(Kokkos_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(view_examples_full PRIVATE -Wall -Wextra -Wpedantic)
endif()
