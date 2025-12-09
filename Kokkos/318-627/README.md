# Kokkos View Examples

This directory contains a compact example that exercises many `Kokkos::View` APIs
illustrated in `core/src/Kokkos_View.hpp` (stride queries, span, operator() /
operator[], impl_map, impl_track, extent_int, and rank-specific access overloads).

Build (example):

1. Create a build directory and run CMake pointing to your Kokkos installation. Example:

```bash
mkdir build && cd build
cmake -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos ..
make -j
./view_examples
```

Notes:
- The example links to `Kokkos::kokkos` (requires a Kokkos installation that exports CMake targets).
- To enable debug bounds checks you must build with the appropriate Kokkos configuration or define
  `-DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK=ON` when building Kokkos (the library CMake that produced
  the Kokkos you link against controls the debug macros used here).
