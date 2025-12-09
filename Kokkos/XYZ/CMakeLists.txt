cmake_minimum_required(VERSION 3.20)
project(MySimulation CXX)

# This tells CMake to look for the Kokkos package automatically
find_package(Kokkos REQUIRED)

# This defines your executable
add_executable(sajid main2.cpp)

# This links the libraries (handling -I, -L, -l, and -fopenmp for you)
target_link_libraries(sajid PRIVATE Kokkos::kokkos)
