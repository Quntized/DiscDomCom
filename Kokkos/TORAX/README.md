# ToraxKokkos: High-Performance Fusion Transport Solver

**ToraxKokkos** is a prototype 1D Tokamak Transport simulator designed for high-performance computing (HPC) and machine learning integration.

It implements a hybrid architecture:
* **Backend:** **C++ / Kokkos** for performance portability (runs on CPU, NVIDIA GPUs, AMD GPUs via OpenMP/CUDA/HIP).
* **Bindings:** **nanobind** for efficient, zero-copy communication between C++ and Python.
* **Frontend:** **Python** for easy configuration, steering, and visualization.

## 🚀 Features

* **1D Radial Transport Solver:** Solves the heat diffusion equation $\frac{\partial T}{\partial t} = \nabla \cdot (n \chi \nabla T) + S$.
* **Kokkos Accelerated:** Parallel execution of the finite difference solver (supports `Kokkos::parallel_for`).
* **Python Interface:** Fully steerable from Python. Pass `numpy` arrays directly to the C++ engine.
* **ML-Ready:** Designed to allow Python-based Neural Networks to infer transport coefficients ($\chi$) and feed them into the C++ solver in real-time.

## 📂 Project Structure

```text
.
├── ToraxKernels.hpp     # The Core Physics Engine (C++ / Kokkos)
├── bindings.cpp         # The Glue Code (Nanobind definitions)
├── CMakeLists.txt       # Build System Configuration
├── run_torax.py         # Python Simulation Script
└── README.md            # Documentation
```
Prerequisites

To build this project, you need:

    C++ Compiler supporting C++17 or C++20 (e.g., g++, clang).

    CMake (Version 3.14+).

    Python 3.x with development headers (e.g., python3-dev).

    Kokkos (If not installed globally, CMake will try to find it; ensure environment is set up).

🔨 Installation & Build

    Clone or Navigate to the directory:
    Bash

cd ~/DDC/Kokkos/TORAX

Create a Build Directory:
Bash

mkdir build
cd build

Configure with CMake: Note: It is recommended to point CMake to your specific Python environment to avoid header issues.
Bash

cmake -DPython_EXECUTABLE=$(which python3) ..

Compile:
Bash

make

Install (Copy Library): Copy the generated shared object file (.so) back to the main directory so Python can import it.
Bash

    cp torax_kokkos*.so ..
    cd ..

🏃 Usage

Run the simulation using the provided Python script. The script initializes the Kokkos backend, sets physics parameters, evolves the system, and plots the result.
Bash

python3 run_torax.py

Example Output
Plaintext

Kokkos Backend Initialized!
Starting Simulation...
Step 0 complete
...
Step 10000 complete
Simulation finished in 0.0313 seconds
Plot saved to: /home/user/DDC/Kokkos/TORAX/simulation_result.png
Kokkos Finalized Cleanly.

🧠 Python API Reference

The library exposes the module torax_kokkos.
TransportSolver Class
Python

import torax_kokkos as tk

# Initialize Runtime
tk.init()

# Create Solver (Grid Points, Time Step [s], Minor Radius [m])
solver = tk.TransportSolver(100, 1e-4, 2.0)

# Set Initial Conditions (Core Temp [keV], Edge Temp [keV])
# Positional arguments: T_core, T_edge
solver.set_initial_condition(10.0, 0.1)

# Set Transport Model (Constant Chi [m^2/s])
# Positional arguments: chi_coeff
solver.set_transport_model(1.5)

# Advance one time step
solver.step()

# Retrieve Profile (returns list of floats)
temp_profile = solver.get_temperature()

# Cleanup
tk.finalize()

🔮 Roadmap

    [x] Basic Explicit Solver implementation.

    [x] Nanobind Python Bindings.

    [ ] ML Integration: Bind set_transport_model to accept full Numpy arrays for spatially varying χ.

    [ ] Physics Upgrade: Add density evolution and source terms.

    [ ] GPU Validation: Test on CUDA backend.
