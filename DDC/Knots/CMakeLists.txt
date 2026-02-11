cmake_minimum_required(VERSION 3.20)
project(FusionThesisProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(DDC REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(sajid knots.cpp)
target_link_libraries(sajid PRIVATE 
    DDC::core
    Kokkos::kokkos
)
