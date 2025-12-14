#ifndef VMEC_WRITER_HPP
#define VMEC_WRITER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

// Struct to hold the "Non-Shape" Physics Parameters
struct VmecConfig {
    // Grid & Resolution
    std::string mgrid_file = "NONE";
    bool lfreeb = false;
    double delt = 1.0;
    int nfp = 5;         // Number of Field Periods (5 for W7-X, 1 for Tokamak)
    int ncurr = 1;       // 0 = Fixed Pressure, 1 = Fixed Current
    
    // Fourier Resolution
    int mpol = 12;
    int ntor = 12;
    int nzeta = 36;
    
    // Iteration Control
    int ns_array = 99;
    double ftol = 1.0e-12;
    int niter = 1000;
    int nstep = 100;
    int nvacskip = 6;
    
    // Physics / Plasma State
    double gamma = 0.0;
    double curtor = 0.0;       // Toroidal Current
    double phiedge = -1.740;   // Total Magnetic Flux (Webers)
    
    // Profiles (Power Series)
    std::vector<double> am = {16.6e4, -16.6e4}; // Mass/Pressure profile
    std::vector<double> ac = {0.0};             // Current profile
    
    // Magnetic Axis Initial Guess (R, Z)
    // These help VMEC find the center quickly
    double raxis_cc = 5.6343;
    double raxis_cs = 0.35209;
    double zaxis_cs = -0.0000; // usually 0 for stellarator symmetry
    double zaxis_cc = -0.29578; // Broken symmetry term? (Check manual)
};

// The Writer Function
void write_vmec_file(const std::string& filename, 
                     const VmecConfig& config,
                     Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> R_coeffs,
                     Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> Z_coeffs,
                     int N_fft) {
    
    std::ofstream file(filename);
    file.precision(6);
    file << std::scientific;

    file << "&INDATA\n";
    file << "! --- Control Parameters ---\n";
    file << "  MGRID_FILE = '" << config.mgrid_file << "'\n";
    file << "  LFREEB = " << (config.lfreeb ? "T" : "F") << "\n";
    file << "  DELT = " << config.delt << "\n\n";

    file << "! --- Geometry & Resolution ---\n";
    file << "  NFP = " << config.nfp << "\n";
    file << "  NCURR = " << config.ncurr << "\n";
    file << "  MPOL = " << config.mpol << "\n";
    file << "  NTOR = " << config.ntor << "\n";
    file << "  NZETA = " << config.nzeta << "\n\n";

    file << "! --- Iteration Settings ---\n";
    file << "  NS_ARRAY = " << config.ns_array << "\n";
    file << "  FTOL_ARRAY = " << config.ftol << "\n";
    file << "  NITER_array = " << config.niter << "\n";
    file << "  NSTEP = " << config.nstep << "\n";
    file << "  NVACSKIP = " << config.nvacskip << "\n\n";

    file << "! --- Physics Values ---\n";
    file << "  GAMMA = " << config.gamma << "\n";
    file << "  CURTOR = " << config.curtor << "\n";
    file << "  PHIEDGE = " << config.phiedge << "\n";
    
    file << "\n! --- Profiles (Power Series) ---\n";
    file << "  PMASS_TYPE = 'power_series'\n";
    file << "  AM = ";
    for(auto val : config.am) file << val << " ";
    file << "\n";

    file << "  PCURR_TYPE = 'power_series'\n";
    file << "  AC = ";
    for(auto val : config.ac) file << val << " ";
    file << "\n";

    file << "\n! --- Axis Guess ---\n";
    file << "  RAXIS_CC = " << config.raxis_cc << "\n";
    // If you have non-zero CS terms for axis, output them too:
    // file << "  ZAXIS_CS = " << config.zaxis_cs << "\n"; 
    
    file << "\n! --- Boundary Shape (From KokkosFFT) ---\n";
    // Loop through FFT results to write RBC/ZBS
    // NOTE: This writes n=0 (Axisymmetric) components.
    // If you want full 3D shaping, you need a 2D FFT.
    int max_m = config.mpol - 1; 

    for(int m=0; m <= max_m; m++) {
        double norm = (m == 0) ? (1.0/N_fft) : (2.0/N_fft);
        double rbc = R_coeffs(m).real() * norm;
        double zbs = Z_coeffs(m).imag() * norm;

        // Thresholding to keep file clean
        if(std::abs(rbc) > 1e-8)
            file << "  RBC(0," << m << ") = " << rbc << "\n";
        
        if(std::abs(zbs) > 1e-8)
            file << "  ZBS(0," << m << ") = " << zbs << "\n";
    }
    
    file << "/\n"; // End of Namelist
    file.close();
    std::cout << "  -> Wrote file: " << filename << "\n";
}

#endif
