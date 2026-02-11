cmake_minimum_required(VERSION 3.15)
project(MyDDCApp VERSION 1.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find DDC package (installed)
find_package(DDC REQUIRED)

# Create your executable
add_executable(myapp main.cpp)

# Link against DDC
target_link_libraries(myapp PRIVATE DDC::core)

# Optional: Enable warnings
target_compile_options(myapp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Optional: Set output directory
set_target_properties(myapp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
