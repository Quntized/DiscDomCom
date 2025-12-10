#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>

// A "Physics" function that calculates the sum.
// Notice it asks for a "const" View. It promises not to change the data.
// The "Converting Constructor" makes it possible to pass a regular View here.
KOKKOS_INLINE_FUNCTION
double calculate_sum(Kokkos::View<const double*> data, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += data(i);
        
        // UNCOMMENTING THE NEXT LINE WOULD CAUSE A COMPILE ERROR
        // data(i) = 0.0; // Error: assignment of read-only location
    }
    return sum;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int N = 10;

        // 1. CREATE ORIGINAL VIEW (Read-Write)
        // This represents our simulation state (e.g., Temperature).
        Kokkos::View<double*> temperature("Temperature", N);

        // Fill with some data
        Kokkos::parallel_for("InitData", N, KOKKOS_LAMBDA(const int i) {
            temperature(i) = i * 1.5;
        });

        // -------------------------------------------------------------
        // 2. THE CONVERTING CONSTRUCTOR IN ACTION
        // -------------------------------------------------------------
        // We are assigning a View<double*> to a View<const double*>.
        // These are DIFFERENT types! 
        //
        // The compiler looks for a constructor in View<const double*> that
        // accepts View<double*>. It finds the templated constructor 
        // we analyzed, checks "is_constructible" (Is double* -> const double* valid?),
        // returns true, and allows this line to compile.
        Kokkos::View<const double*> read_only_ref = temperature;
        std::cout<<"Pointer memory address temperature :  "<<temperature.data()<<std::endl;
        std::cout<<"Pointer memory address read_only_ref(non const to const):  "<<read_only_ref.data()<<std::endl;

        std::cout << "Original View Label:  " << temperature.label() << "\n";
        std::cout << "Converted View Label: " << read_only_ref.label() << "\n\n";

        // 3. PROVE DATA SHARING
        // The converted view points to the exact same memory.
        std::cout << "Modifying index 0 via the ORIGINAL view...\n";
        Kokkos::parallel_for("Modify", 1, KOKKOS_LAMBDA(const int i) {
            temperature(0) = 999.9; 
        });
        Kokkos::fence(); // Wait for GPU to finish

        // We read it back through the READ-ONLY view.
        // We need a HostMirror to print on CPU
        auto host_mirror = Kokkos::create_mirror_view(read_only_ref);
        Kokkos::deep_copy(host_mirror, read_only_ref);

        std::cout << "Value seen by Read-Only View at index 0: " 
                  << host_mirror(0) << " (Should be 999.9)\n\n";

        // 4. INVALID CONVERSION (The Safety Check)
        // If you try to go the other way (Const -> Non-Const), 
        // the "std::enable_if" in the constructor will fail, and this 
        // line would cause a massive compilation error.
        
        // Kokkos::View<double*> illegal = read_only_ref; // <--- COMPILATION ERROR!

    } // Destructors run here
    Kokkos::finalize();
    return 0;
}
