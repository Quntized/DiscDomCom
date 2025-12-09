Kokkos View Examples (full)

This folder contains a comprehensive example `view_examples_full` demonstrating many
of the public `Kokkos::View` APIs shown in `core/src/Kokkos_View.hpp`.

Files:
- `CMakeLists.txt` - build the example with a Kokkos installation.
- `view_examples_full.cpp` - the example program.

Build instructions (from workspace root):

1) Create a build directory and configure (pass your Kokkos install location if needed):

```bash
mkdir -p examples/view_examples_full/build
cd examples/view_examples_full/build
cmake -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos ..
```

If you built Kokkos in-tree, point `-DKokkos_DIR` at the `build` tree that contains `KokkosConfig.cmake`.

2) Build and run:

```bash
make -j
./view_examples_full
```

Notes and highlights:
- Demonstrates rank-0, rank-1, rank-2, rank-3 views.
- Shows `operator()` and `operator[]` usage, `access(...)` overloads with trailing zeros,
  `span()`, `stride()`, `is_allocated()`, `data()`, `impl_track()`.
- Uses `MDRangePolicy` for multi-dimensional parallel initialization and `create_mirror_view_and_copy`
  to bring data back to host for printing.

If you want, I can now try to configure, build, and run this example here (I will need the path
used for your installed Kokkos if CMake cannot find it automatically).