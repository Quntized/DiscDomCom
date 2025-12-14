#include "plasma_lib.hpp"
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <fstream>
#include <cmath>
#include <iostream>
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
PlasmaGenerator::PlasmaGenerator() {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }
}

void PlasmaGenerator::set_config(const PlasmaConfig& config) {
    this->cfg = config;
}

void PlasmaGenerator::generate_vmec_file(const std::string& filename) {
    std::cout << "[Lib] Generating shape on GPU...\n";
    int N = cfg.grid_points;
    Kokkos::View<double*, ExecutionSpace> R_real("R", N);
    Kokkos::View<double*, ExecutionSpace> Z_real("Z", N);
    PlasmaConfig local_cfg = this->cfg; 
    Kokkos::parallel_for("GenerateShape", N, KOKKOS_LAMBDA(const int i) {
        double theta = 2.0 * M_PI * i / N;
        R_real(i) = local_cfg.major_radius + 
                    local_cfg.minor_radius * cos(theta + local_cfg.triangularity * sin(theta));
        Z_real(i) = local_cfg.elongation * local_cfg.minor_radius * sin(theta);
    });
    Kokkos::fence();
    Kokkos::View<Kokkos::complex<double>*, ExecutionSpace> R_freq("R_freq", N/2+1);
    Kokkos::View<Kokkos::complex<double>*, ExecutionSpace> Z_freq("Z_freq", N/2+1);
    KokkosFFT::Plan plan_R(ExecutionSpace(), R_real, R_freq, KokkosFFT::Direction::forward, 0, N);
    KokkosFFT::Plan plan_Z(ExecutionSpace(), Z_real, Z_freq, KokkosFFT::Direction::forward, 0, N);
    KokkosFFT::execute(plan_R, R_real, R_freq);
    KokkosFFT::execute(plan_Z, Z_real, Z_freq);
    auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_freq);
    auto Z_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Z_freq);
    std::vector<double> rbc_vec, zbs_vec;
    for(int m=0; m < cfg.fourier_modes; m++) {
        double norm = (m==0) ? (1.0/N) : (2.0/N);
        rbc_vec.push_back(R_host(m).real() * norm);
        zbs_vec.push_back(Z_host(m).imag() * norm);
    }
    write_to_disk(filename, rbc_vec, zbs_vec);
}
void PlasmaGenerator::write_to_disk(const std::string& fname, 
                                    const std::vector<double>& rbc, 
                                    const std::vector<double>& zbs) {
    std::ofstream file(fname);
    file << "&INDATA\n";
    file << "  NFP = " << cfg.nfp << "\n";
    file << "  PHIEDGE = " << cfg.flux << "\n";
    file << "  MPOL = " << cfg.fourier_modes << "\n";
    
    // Write the Modes
    for(size_t m=0; m<rbc.size(); m++) {
        if(std::abs(rbc[m]) > 1e-9) file << "  RBC(0," << m << ") = " << rbc[m] << "\n";
        if(std::abs(zbs[m]) > 1e-9) file << "  ZBS(0," << m << ") = " << zbs[m] << "\n";
    }
    file << "/\n";
    file.close();
    std::cout << "[Lib] File written: " << fname << "\n";
}
