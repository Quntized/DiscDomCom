cmake_minimum_required(VERSION 3.20)
project(KokkosFFTDemo LANGUAGES CXX)

find_package(Kokkos REQUIRED)
find_package(KokkosFFT REQUIRED)

add_executable(sh shape.cpp)
target_link_libraries(sh PRIVATE Kokkos::kokkos KokkosFFT::fft)
