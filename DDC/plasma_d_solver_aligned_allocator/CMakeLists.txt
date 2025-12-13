cmake_minimum_required(VERSION 3.20)
project(mine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
find_package(DDC REQUIRED)
add_executable(my plasma_d_solver.cpp)
target_link_libraries(my PRIVATE DDC::core OpenMP::OpenMP_CXX)

