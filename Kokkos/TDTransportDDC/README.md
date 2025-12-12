# TDTransportDDC

This directory provides a modular, term-by-term implementation of the time-dependent neutron transport equation.

Structure
- `include/TDTransportDDC` : operator interfaces and headers for each term
- `src` : implementations and pybind11 bindings
- `examples` : small native and python examples

Design notes
- Each major term from the equation is implemented in its own pair of header/source files:
  - `TimeDerivative` (1/v * d/dt)
  - `StreamingOperator` (Omega·∇)
  - `TotalXSOperator` (Σ_t φ)
  - `ScatteringOperator` (∫ Σ_s φ)
  - `FissionOperator` (χ ν Σ_f)
  - `ExternalSource` (s)
- `TransportEquation` composes these modules and performs a backward-Euler time step.
- The code provides host-array variants (std::vector-based) which are exposed to Python via pybind11.
- Hook points for DDC: the `StreamingOperator` contains a placeholder where mesh connectivity and DDC field operations should be used. If you place the DDC repository in the project root as `ddc/` CMake will add it as a subdir and link it into the core library.

Build instructions
1. Ensure Kokkos and pybind11 are available to CMake. Optionally put the DDC sources into `TDTransportDDC/ddc/`.

```bash
mkdir build && cd build
cmake .. -DKokkos_DIR=/path/to/kokkos/cmake -DCMAKE_BUILD_TYPE=Release
make -j
```

Python usage

```python
import tdtransport_ddc
solver = tdtransport_ddc.TransportEquation(10,10,1,2,8)
# set arrays (flattened group-major G*ncells)
solver.set_total_xs([...])
solver.set_external_source([...])
solver.step_time(10, 0.01)
phi0 = solver.get_phi(0)
```

Next steps (recommended for thesis-grade work)
- Implement the streaming discretization using DDC field/mesh primitives and Kokkos parallel loops.
- Add Level-Symmetric S_N quadrature and angular-to-scalar projection.
- Implement acceleration (DSA) and implicit multigroup solvers.
- Add verification against analytical/manufactured solutions and benchmark problems (slab, Gauss-Seidel, etc.).
