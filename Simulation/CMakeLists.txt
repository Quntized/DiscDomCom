cmake_minimum_required(VERSION 3.20)
project(FusionTransportDDC LANGUAGES CXX)

# Dependencies
find_package(DDC REQUIRED)      # The core library
find_package(Kokkos REQUIRED)   # For Parallelism (CPU/GPU)
find_package(MPI REQUIRED)      # For Distributed Computing
find_package(HDF5 REQUIRED)     # For I/O

# Compiler Flags for C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Executable
add_executable(fusion_sim 
    src/main.cpp 
    src/transport_solver.cpp
)

# Link Libraries
target_link_libraries(fusion_sim 
    PRIVATE 
    DDC::ddc 
    Kokkos::kokkos 
    MPI::MPI_CXX
    HDF5::HDF5
)

# GPU Optimization flags (if CUDA enabled)
if(Kokkos_ENABLE_CUDA)
    target_compile_options(fusion_sim PRIVATE --expt-extended-lambda)
endif()
