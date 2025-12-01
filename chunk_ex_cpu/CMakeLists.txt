cmake_minimum_required(VERSION 3.20)

project(sajid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DDC REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(sajid chunk_example.cpp)
target_link_libraries(sajid PRIVATE DDC::core Kokkos::kokkos)
