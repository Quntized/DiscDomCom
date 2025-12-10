#ifndef TORAX_KERNELS_HPP
#define TORAX_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include <cmath>

// Define the execution space (GPU or CPU based on compilation)
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;

// Define Views (Arrays) similar to JAX arrays
using View1D = Kokkos::View<double*, MemSpace>;
using View1D_Host = Kokkos::View<double*>::HostMirror;

class TransportSolver {
public:
    int n_grid;
    View1D T_e;       // Electron Temperature
    View1D chi_e;     // Diffusivity coefficient
    View1D source;    // Heating source
    View1D r;         // Radial grid
    double dt;
    double dr;

    // Constructor: Initialize grids
    TransportSolver(int n, double time_step, double radius) 
        : n_grid(n), dt(time_step), dr(radius / (n - 1)) {
        
        // Allocate memory on Device (GPU/CPU)
        T_e = View1D("T_e", n);
        chi_e = View1D("chi_e", n);
        source = View1D("source", n);
        r = View1D("r", n);

        // Initialize grid on Host then copy to Device
        View1D_Host r_host = Kokkos::create_mirror_view(r);
        for (int i = 0; i < n; i++) {
            r_host(i) = i * dr;
        }
        Kokkos::deep_copy(r, r_host);
    }

    // Initialize profiles (Initial Condition)
    void set_initial_condition(double T_core, double T_edge) {
        // Parallel loop on GPU to set values
        auto T = T_e;
        auto grid_r = r;
        double radius = dr * (n_grid - 1);
        
        Kokkos::parallel_for("InitProfile", Kokkos::RangePolicy<ExecSpace>(0, n_grid), 
            KOKKOS_LAMBDA(const int i) {
                // Parabolic profile: T = T_edge + (T_core - T_edge)*(1 - (r/a)^2)
                double normalized_r = grid_r(i) / radius;
                T(i) = T_edge + (T_core - T_edge) * (1.0 - normalized_r * normalized_r);
            });
    }

    // Set Transport Coefficients (The Physics)
    void set_transport_model(double chi_coeff) {
        auto chi = chi_e;
        Kokkos::parallel_for("SetChi", Kokkos::RangePolicy<ExecSpace>(0, n_grid), 
            KOKKOS_LAMBDA(const int i) {
                chi(i) = chi_coeff; // Constant Bohm-like transport for now
            });
    }

    // The Main Physics Step: Explicit Finite Difference Solver
    // dT/dt = (1/r) * d/dr ( r * chi * dT/dr ) + Source
    void step() {
        // Create temporary view for the new state
        View1D T_new("T_new", n_grid);
        
        // Capture class members for lambda
        auto T_old = T_e;
        auto chi = chi_e;
        auto src = source;
        auto rad = r;
        double _dt = dt;
        double _dr = dr;
        int _n = n_grid;

        Kokkos::parallel_for("SolverStep", Kokkos::RangePolicy<ExecSpace>(1, _n - 1), 
            KOKKOS_LAMBDA(const int i) {
                // Finite Difference Discretization
                double r_i = rad(i);
                
                // Flux at i+1/2 and i-1/2
                double grad_plus = (T_old(i+1) - T_old(i)) / _dr;
                double grad_minus = (T_old(i) - T_old(i-1)) / _dr;
                
                double flux_plus = chi(i) * grad_plus;
                double flux_minus = chi(i-1) * grad_minus; // Simplified staggered grid
                
                // Diffusion term: (1/r) * div(flux)
                double diffusion = (1.0 / r_i) * ( (r_i + 0.5*_dr)*flux_plus - (r_i - 0.5*_dr)*flux_minus ) / _dr;
                
                // Time update
                T_new(i) = T_old(i) + _dt * (diffusion + src(i));
            });
            
        // Apply Boundary Conditions
        Kokkos::parallel_for("BoundaryConditions", Kokkos::RangePolicy<ExecSpace>(0, 1), 
            KOKKOS_LAMBDA(const int i) {
                T_new(0) = T_new(1); // Zero gradient at core
                T_new(_n - 1) = 0.1; // Fixed edge temperature
            });

        // Swap pointers (update state)
        // Note: Deep copy for simplicity here, pointer swap is faster in optimized code
        Kokkos::deep_copy(T_e, T_new);
    }

    // Helper to get data back to Python (Host)
    std::vector<double> get_temperature() {
        View1D_Host T_host = Kokkos::create_mirror_view(T_e);
        Kokkos::deep_copy(T_host, T_e);
        
        std::vector<double> result(n_grid);
        for (int i = 0; i < n_grid; i++) result[i] = T_host(i);
        return result;
    }
};

#endif
