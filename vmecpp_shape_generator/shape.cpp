#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "vmec_writer.hpp" // Include the writer struct we just made
#include <cmath>

// Define standard types
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using RealView       = Kokkos::View<double*, ExecutionSpace>;
using ComplexView    = Kokkos::View<Kokkos::complex<double>*, ExecutionSpace>;

int main(int argc, char* argv[]) {
    // 1. Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);
    std::cout << "--- Kokkos -> VMEC Shape Generator ---\n";
    std::cout << "Running on: " << ExecutionSpace::name() << "\n";

    // ---------------------------------------------------------
    // STEP 1: Define Physics Parameters for the Shape
    // ---------------------------------------------------------
    // We will generate a "D-Shaped" Cross Section
    double R0 = 5.5;     // Major Radius (Matches your RAXIS roughly)
    double a  = 0.5;     // Minor Radius
    double kappa = 1.7;  // Elongation
    double delta = 0.4;  // Triangularity

    int N = 64; // Grid Resolution
    RealView R_real("R_real", N);
    RealView Z_real("Z_real", N);

    // ---------------------------------------------------------
    // STEP 2: The Kokkos Kernel (Generate Points)
    // ---------------------------------------------------------
    Kokkos::parallel_for("FillShape", Kokkos::RangePolicy<ExecutionSpace>(0, N), 
        KOKKOS_LAMBDA(const int i) {
            double theta = 2.0 * M_PI * i / N;
            
            // Parametric D-Shape Equations
            R_real(i) = R0 + a * cos(theta + delta * sin(theta));
            Z_real(i) = kappa * a * sin(theta);
        });

    Kokkos::fence(); // Ensure GPU finishes writing

    // ---------------------------------------------------------
    // STEP 3: The Math Engine (KokkosFFT)
    // ---------------------------------------------------------
    ComplexView R_coeffs("R_coeffs", N / 2 + 1);
    ComplexView Z_coeffs("Z_coeffs", N / 2 + 1);

    // Create Plans (Explicit arguments: Space, In, Out, Direction, Axis, Length)
    KokkosFFT::Plan plan_R(ExecutionSpace(), R_real, R_coeffs, KokkosFFT::Direction::forward, 0, N);
    KokkosFFT::Plan plan_Z(ExecutionSpace(), Z_real, Z_coeffs, KokkosFFT::Direction::forward, 0, N);

    // Execute FFT
    KokkosFFT::execute(plan_R, R_real, R_coeffs);
    KokkosFFT::execute(plan_Z, Z_real, Z_coeffs);
    
    Kokkos::fence();

    // ---------------------------------------------------------
    // STEP 4: Configuration & Output (Using vmec_writer.hpp)
    // ---------------------------------------------------------
    // Copy results back to CPU for file writing
    auto R_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R_coeffs);
    auto Z_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Z_coeffs);

    // Create the Configuration Object
    VmecConfig config;

    // TWEAK YOUR PARAMETERS HERE!
    // ---------------------------
    config.nfp = 5;          // W7-X style 5-period symmetry
    config.phiedge = -1.74;  // Magnetic Flux
    config.mpol = 12;        // Number of poloidal modes
    
    // Set the Magnetic Axis guess to match our R0
    config.raxis_cc = R0; 
    config.raxis_cs = 0.0;

    // Write the file
    std::cout << "Writing VMEC input file...\n";
    write_vmec_file("input.test_shape", config, R_host, Z_host, N);

    return 0;
}
