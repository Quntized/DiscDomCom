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
```
import numpy as np
import matplotlib.pyplot as plt
import diffusion_kokkos
import time

# --- Parameters ---
D       = 0.01   # Diffusivity
L       = 1.0    # Domain length
Nx      = 101    # Grid points
t_final = 1.0    # Run for 1.0 second

# --- Stability Calculation ---
dx = L / (Nx - 1)
# We choose dt dynamically to be safe (alpha = 0.4)
dt_stable = 0.4 * (dx * dx) / D 

print(f"--- Setup ---")
print(f"dx: {dx:.6f}")
print(f"Calculated stable dt: {dt_stable:.6f}")

# --- Run Simulation ---
print("Running Kokkos simulation...")
start_time = time.time()

# We pass our calculated safe dt
result = diffusion_kokkos.simulate(
    diffusivity=D, 
    length=L, 
    Nx=Nx, 
    dt=dt_stable, 
    t_final=t_final, 
    bc="dirichlet"
)

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.4f} seconds.")

# --- Plotting (Save to file) ---
x = np.linspace(0, L, Nx)

plt.figure(figsize=(10, 6))
plt.plot(x, result, label=f't={t_final}s')
plt.title(f"1D Diffusion (Kokkos + Python)\nN={Nx}, dt={dt_stable:.2e}")
plt.xlabel("Position")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# FIX: Save the file instead of trying to show a window
filename = "diffusion_result.png"
plt.savefig(filename)
print(f"Plot saved to '{filename}'")
```

Notes

- The solver uses an explicit forward-Euler finite-difference discretization. For Dirichlet/Neumann BCs the scheme is stable when alpha = D*dt/dx^2 <= 0.5 approximately; the code warns if alpha>0.5. For larger dt, consider implementing an implicit scheme (e.g., Crank-Nicolson).
- The module initializes Kokkos automatically on the first call to `simulate(...)`. We intentionally do not finalize Kokkos to allow multiple calls from Python during the same process. If you want strict lifetime control, wrap calls in a separate process or extend the module with explicit init/finalize functions.

