#include "plasma_lib.hpp"
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <array>
#include <map>
#include <algorithm> 

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

// 1. Constructor
PlasmaLib::PlasmaLib() {
    if (!Kokkos::is_initialized()) Kokkos::initialize();
}

// 2. Setter
void PlasmaLib::set_config(const PlasmaConfig& config) {
    this->cfg = config;
}

// 3. Generator
void PlasmaLib::generate_file(const std::string& filename) {
    std::cout << "[Lib] Generating Plasma Surface (" << cfg.resolution << "x" << cfg.resolution << ")...\n";

    // --- PART A: Surface Generation ---
    int N = cfg.resolution;
    Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace> R_surf("R_surf", N, N);
    Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace> Z_surf("Z_surf", N, N);

    PlasmaConfig local_cfg = this->cfg; 

    Kokkos::parallel_for("GenerateLCFS", 
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0,0}, {N, N}),
        KOKKOS_LAMBDA(const int j, const int i) {
            double zeta  = 2.0 * M_PI * j / N; 
            double theta = 2.0 * M_PI * i / N; 
            
            double eff_kappa = local_cfg.kappa + local_cfg.torsion * cos(zeta);
            double shift_R   = local_cfg.axis_shift_amp * cos(zeta);
            double shift_Z   = -local_cfg.axis_shift_amp * sin(zeta);

            R_surf(j, i) = (local_cfg.R0 + shift_R) + 
                           local_cfg.a * cos(theta + local_cfg.delta * sin(theta));
            Z_surf(j, i) = shift_Z + (local_cfg.a * eff_kappa * sin(theta));
        });
    Kokkos::fence();

    // --- PART B: FFT ---
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutRight, ExecutionSpace> R_hat("R_hat", N, N/2+1);
    Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutRight, ExecutionSpace> Z_hat("Z_hat", N, N/2+1);

    std::array<int, 2> axes = {0, 1}; 
    std::array<unsigned long, 2> fft_shape = {(unsigned long)N, (unsigned long)N};

    KokkosFFT::Plan plan_R(ExecutionSpace(), R_surf, R_hat, KokkosFFT::Direction::forward, axes, fft_shape);
    KokkosFFT::Plan plan_Z(ExecutionSpace(), Z_surf, Z_hat, KokkosFFT::Direction::forward, axes, fft_shape);
    
    KokkosFFT::execute(plan_R, R_surf, R_hat);
    KokkosFFT::execute(plan_Z, Z_surf, Z_hat);
    Kokkos::fence();

    auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_hat);
    auto Z_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Z_hat);

    // --- PART C: Filter Modes ---
    std::map<std::pair<int,int>, double> rbc_map;
    std::map<std::pair<int,int>, double> zbs_map;
    int detected_max_m = 0, detected_max_n = 0;

    for(int j = 0; j < N; j++) {
        for(int i = 0; i < N/2 + 1; i++) {
            int m = i; 
            int n = (j <= N/2) ? j : (j - N); 
            
            double norm = 1.0 / (N * N);
            if (m > 0 || n != 0) norm *= 2.0;

            double r_val = R_host(j, i).real() * norm;
            double z_val = -Z_host(j, i).imag() * norm; 

            if (std::abs(r_val) > cfg.detect_tol) {
                rbc_map[{n, m}] = r_val;
                if(m > detected_max_m) detected_max_m = m;
                if(std::abs(n) > detected_max_n) detected_max_n = std::abs(n);
            }
            if (std::abs(z_val) > cfg.detect_tol) {
                zbs_map[{n, m}] = z_val;
                if(m > detected_max_m) detected_max_m = m;
                if(std::abs(n) > detected_max_n) detected_max_n = std::abs(n);
            }
        }
    }

    int write_mpol = std::max(cfg.mpol, detected_max_m + 1);
    int write_ntor = std::max(cfg.ntor, detected_max_n);
    write_vmec_format(filename, write_mpol, write_ntor, rbc_map, zbs_map);
}

// 4. The Writer (CRITICAL FIX: PADDING LOGIC)
void PlasmaLib::write_vmec_format(const std::string& fname, int max_m, int max_n, 
                                  const std::map<std::pair<int,int>, double>& rbc,
                                  const std::map<std::pair<int,int>, double>& zbs) {
    std::ofstream file(fname);
    file << std::scientific << std::setprecision(12);
    
    // --- SAFETY PADDING ---
    // This Lambda adds 0.0 until the vector has at least 3 elements.
    auto pad_vector = [](std::vector<double> v) {
        while(v.size() < 3) v.push_back(0.0);
        return v;
    };

    // Pad everything to avoid "Need 3, got 2" error
    std::vector<double> safe_am = pad_vector(cfg.am);
    std::vector<double> safe_ac = pad_vector(cfg.ac);
    std::vector<double> safe_ai = pad_vector(cfg.ai);

    file << "&INDATA\n\n";
    file << "LASYM = " << (cfg.lasym ? "T" : "F") << ",\n";
    file << "NFP = " << cfg.nfp << ",\n";
    file << "MPOL = " << max_m << ",\n";
    file << "NTOR = " << max_n << ",\n";
    file << "NZETA = " << cfg.nzeta << ",\n\n";

    write_array(file, "NS_ARRAY", cfg.ns_array);
    write_array(file, "FTOL_ARRAY", cfg.ftol_array);
    write_array(file, "NITER_ARRAY", cfg.niter_array);
    
    file << "\nPHIEDGE = " << cfg.phiedge << ",\n";
    file << "NCURR = " << cfg.ncurr << ",\n\n";
    
    // Mass/Pressure
    file << "pmass_type = 'two_power',\n";
    write_array(file, "AM", safe_am); // Using padded AM
    file << "PRES_SCALE = " << cfg.pres_scale << ",\n";
    file << "GAMMA = " << cfg.gamma << ",\n";
    file << "SPRES_PED = " << cfg.spres_ped << ",\n\n";

    // Current / Iota
    // We write AC even if NCURR=0 to keep the parser happy, just in case
    file << "PCURR_TYPE = 'two_power',\n";
    write_array(file, "AC", safe_ac); // Using padded AC

    if (cfg.ncurr == 0) {
        file << "piota_type = 'power_series',\n";
        write_array(file, "AI", safe_ai);
    } else {
        file << "CURTOR = " << cfg.curtor << ",\n";
        file << "BLOAT = " << cfg.bloat << ",\n";
    }
    
    file << "\nLFREEB = " << (cfg.lfreeb ? "T" : "F") << ",\n";
    file << "\nNSTEP = " << cfg.nstep << ",\n";
    file << "DELT = " << cfg.delt << ",\n\n";

    file << "! Initial Position.\n";
    write_array(file, "RAXIS_CC(:)", cfg.raxis); 
    write_array(file, "ZAXIS_CS(:)", cfg.zaxis);
    
    file << "\n";
    for (auto const& [key, val] : rbc) {
        file << "RBC(" << std::setw(3) << key.first << "," << std::setw(3) << key.second << ") = " 
             << std::setw(20) << val << "\n";
    }
    file << "\n";
    for (auto const& [key, val] : zbs) {
        file << "ZBS(" << std::setw(3) << key.first << "," << std::setw(3) << key.second << ") = " 
             << std::setw(20) << val << "\n";
    }
    
    file << "/\n";
    file.close();
    std::cout << "[Lib] File " << fname << " written. (Profiles padded to size 3)\n";
}
