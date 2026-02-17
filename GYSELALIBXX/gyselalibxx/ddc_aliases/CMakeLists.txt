# CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(DDC_Heat_Diffusion VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compile options for better warnings and optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find required packages
find_package(DDC REQUIRED)
find_package(Kokkos REQUIRED)

# Create ddc_aliases.hpp header target (interface library)
add_library(ddc_aliases INTERFACE)
target_include_directories(ddc_aliases INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Main executable
add_executable(heat_diffusion_1d main.cpp)

target_link_libraries(heat_diffusion_1d
    PRIVATE
        DDC::DDC
        Kokkos::kokkos
        ddc_aliases
)

# Enable position independent code
set_target_properties(heat_diffusion_1d PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Installation rules (optional)
install(TARGETS heat_diffusion_1d
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  CMake version     : ${CMAKE_VERSION}")
message(STATUS "  System            : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ Compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard      : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  DDC found         : ${DDC_FOUND}")
message(STATUS "  Kokkos found      : ${Kokkos_FOUND}")
message(STATUS "")
