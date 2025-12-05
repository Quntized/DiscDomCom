# 1. Minimum CMake version
cmake_minimum_required(VERSION 3.20)

# 2. Project Name
project(KokkosAssignmentCheck LANGUAGES CXX)

# 3. Set C++ Standard (Kokkos requires C++17 or newer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. Find the Kokkos Package
# This looks for KokkosConfig.cmake in your system paths.
find_package(Kokkos REQUIRED)

# 5. Define the Executable
# Replace "main.cpp" with the actual name of your C++ file.
add_executable(assign_check ptr_memory_address.cpp)

# 6. Link Kokkos
# This links the executable to the Kokkos library and sets up include paths.
target_link_libraries(assign_check PRIVATE Kokkos::kokkos)
