Kokkos 1D Diffusion Solver with Python bindings

This example provides:
- A Kokkos-only explicit 1D diffusion solver implemented in C++ (`diffusion_kokkos_py.cpp`).
- A Python extension module built with `pybind11` named `diffusion_kokkos` exposing `simulate(...)`.
- A small CLI executable `diffusion_kokkos_cli` that runs the same solver from the terminal.

Directory:
- `CMakeLists.txt` - build the Python module and CLI. You must have `pybind11` and `Kokkos` available.
- `diffusion_kokkos_py.cpp` - solver + pybind11 module
- `diffusion_kokkos_cli.cpp` - thin CLI wrapper

Build instructions

1) Install dependencies:
   - Kokkos (built and installed or in-tree). You may already have it in this workspace.
   - pybind11 (system package or via pip). On many systems: `sudo apt install pybind11-dev` or `pip install pybind11` for user installs.

2) Configure and build (example from the repository root):

```bash
mkdir -p examples/diffusion_kokkos_py/build
cd examples/diffusion_kokkos_py/build
# Point to your Kokkos install location if CMake does not find it automatically:
cmake -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos ..
make -j
```

This produces:
- A Python extension module `diffusion_kokkos` (the `.so` / `.pyd`) in the build tree.
- A CLI binary `diffusion_kokkos_cli` in the build tree.

Python usage example

From the build directory (so Python can import the local module):

```python
import numpy as np
import diffusion_kokkos

# parameters
D = 0.1
L = 1.0
Nx = 501
dt = 1e-5
t_final = 0.01

u = diffusion_kokkos.simulate(D, L, Nx, dt, t_final, 'dirichlet')
print('result shape', u.shape)
print('u[::100] =', u[::100])
```

Notes

- The solver uses an explicit forward-Euler finite-difference discretization. For Dirichlet/Neumann BCs the scheme is stable when alpha = D*dt/dx^2 <= 0.5 approximately; the code warns if alpha>0.5. For larger dt, consider implementing an implicit scheme (e.g., Crank-Nicolson).
- The module initializes Kokkos automatically on the first call to `simulate(...)`. We intentionally do not finalize Kokkos to allow multiple calls from Python during the same process. If you want strict lifetime control, wrap calls in a separate process or extend the module with explicit init/finalize functions.

If you want, I can now try to build and run the example here (I will attempt to detect your Kokkos location automatically; otherwise provide `-DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos`).