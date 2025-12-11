cmake_minimum_required(VERSION 3.20)
project(KokkosWrap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)

add_executable(sajid pointer_view_alloc.cpp)
target_link_libraries(sajid PRIVATE Kokkos::kokkos)
