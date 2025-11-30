cmake_minimum_required(VERSION 3.20)
project(DDCTest)

# Use C++17 (Standard for many DDC versions)
set(CMAKE_CXX_STANDARD 17)

# 1. Find the DDC library
find_package(DDC REQUIRED)

# 2. Define your executable
add_executable(ddc_test main.cpp)

# 3. Link DDC to your executable
target_link_libraries(ddc_test PRIVATE DDC::core)

