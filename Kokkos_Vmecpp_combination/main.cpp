#include "plasma_lib.hpp"
#include <iostream>

int main() {
    PlasmaLib lib;
    PlasmaConfig cfg;

    std::cout << "--- Configuring ROBUST W7-X Simulation ---\n";
    
    // 1. Grid & Resolution
    cfg.nfp = 5;
    cfg.nzeta = 36;
    cfg.ns_array = {15, 31, 49};        
    cfg.ftol_array = {1.0e-8, 1.0e-10, 1.0e-12}; 
    cfg.niter_array = {2000, 5000, 10000};
    cfg.delt = 0.5;
    cfg.phiedge = -1.0; 
    
    // 2. Physics
    cfg.ncurr = 0;          // Fixed Iota
    cfg.pres_scale = 5.0e4; // Finite Beta (50 kPa)

    // Profiles:
    // With the new plasma_lib.cpp, these will be auto-padded if you make a mistake.
    // But it's good practice to provide 3 values for 'two_power'.
    cfg.am = {0.0, -1.0, 0.0}; 
    cfg.ac = {0.0, 0.0, 0.0};  

    // Iota uses power_series, so 3 coeffs is fine (polynom: a0 + a1*s + a2*s^2)
    cfg.ai = {0.86, 0.12, 0.0}; 

    // 3. Geometry
    cfg.R0 = 5.5; 
    cfg.a = 0.52;
    cfg.kappa = 1.0;
    cfg.delta = 0.0;
    cfg.torsion = 0.1;
    cfg.axis_shift_amp = 0.05;

    // 4. Axis Guess
    cfg.raxis = { 5.5, 0.15 }; 
    cfg.zaxis = { 0.0, 0.15 };

    // 5. Generate
    cfg.resolution = 256;
    lib.set_config(cfg);
    lib.generate_file("input.w7x_robust");

    return 0;
}
