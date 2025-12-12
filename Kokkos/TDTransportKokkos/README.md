# TDTransportKokkos

Kokkos-based time-dependent neutron transport solver with Python bindings (pybind11).

Build (example):

```bash
mkdir build && cd build
cmake .. -DKokkos_DIR=/path/to/kokkos/cmake -DCMAKE_BUILD_TYPE=Release
make -j
```

Notes:
- Requires Kokkos and pybind11 (CONFIG mode) available to CMake.
- The implementation includes a realistic multi-group, multi-angle formulation structure.
- The sweep implementation provided is a straightforward upwind serial sweep; replace with a production sweep and acceleration (DSA, GMRES, etc.) for high-performance use.

Python usage:

```python
import tdtransport
solver = tdtransport.TransportSolver(10,10,1,2,8)
# prepare arrays (flattened by group*ncells)
# solver.set_total_xs([...])
# solver.set_external_source([...])
solver.run_time_steps(10, 0.01)
phi = solver.get_scalar_flux(0)
```
