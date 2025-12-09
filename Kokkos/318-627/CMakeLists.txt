cmake_minimum_required(VERSION 3.10)
project(kokkos_view_examples LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(view_examples view_examples.cpp)
target_link_libraries(view_examples Kokkos::kokkos)
set_target_properties(view_examples PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
