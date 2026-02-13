cmake_minimum_required(VERSION 3.15)
project(Sajid LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(DDC REQUIRED)
add_executable(sajid main.cpp)
target_link_libraries(sajid PRIVATE DDC::core)

