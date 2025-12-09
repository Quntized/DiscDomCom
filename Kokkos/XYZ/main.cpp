#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
    // 1. Initialize Kokkos (Sets up the execution environment, e.g., GPU context or CPU threads)
    Kokkos::initialize(argc, argv);
    {
        // Define N: The size of our plasma grid points
        int N = 10; 

        // 2. DATA MANAGEMENT (The "View")
        // Instead of std::vector or raw arrays, we use Kokkos::View.
        // "Device" means this memory lives where the computation happens (e.g., GPU memory).
        Kokkos::View<double*> x("x_vector", N);
        Kokkos::View<double*> y("y_vector", N);

        // 3. PARALLEL DISPATCH (The "Pattern")
        // Initialize x to 1.0 and y to 2.0
        // We do NOT write a raw 'for' loop. We ask Kokkos to parallelize it.
        Kokkos::parallel_for("Init_Arrays", N, KOKKOS_LAMBDA(const int i) {
            x(i) = 1.0;
            y(i) = 2.0;
        });

        double a = 0.5; // Scalar constant

        // 4. THE PHYSICS CALCULATION
        // Calculate Y = a*X + Y parallelized across the device.
        Kokkos::parallel_for("SAXPY_Computation", N, KOKKOS_LAMBDA(const int i) {
            y(i) = a * x(i) + y(i);
            
            // Note: We cannot easily print from inside a GPU kernel, 
            // but for OpenMP/CPU it works.
            // printf("Index %d calculated: %f\n", i, y(i));
        });

        // Fence: Wait for all parallel jobs to finish before moving on
        Kokkos::fence();

        // 5. GETTING DATA BACK (Deep Copy)
        // If 'y' is on a GPU, we can't read it on the CPU directly.
        // We create a "Host Mirror" to copy the data back to main RAM.
        auto y_host = Kokkos::create_mirror_view(y); 
        Kokkos::deep_copy(y_host, y); // Copy Device -> Host

        // Print result from the CPU
        printf("Result at index 0: %f\n", y_host(0)); 
    }
    
    // 6. Finalize: Clean up hardware resources
    Kokkos::finalize();
}
