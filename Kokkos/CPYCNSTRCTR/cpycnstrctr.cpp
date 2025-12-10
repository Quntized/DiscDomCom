#include <Kokkos_Core.hpp>
#include <iostream>

void example_casts() {
    // 1. ORIGINAL (Managed, Read-Write)
    // "a" owns the memory. When "a" dies, memory is freed.
    Kokkos::View<double*> a("View_A", 10);

    // Initialize data
    Kokkos::parallel_for("Init", 10, KOKKOS_LAMBDA(const int i) {
        a(i) = i * 1.5; 
    });

    // ---------------------------------------------------------
    // SCENARIO 1: Const-Correctness
    // We want to pass "a" to a function that shouldn't modify it.
    // The constructor `View(const View& other)` handles this compatible cast.
    // ---------------------------------------------------------
    Kokkos::View<const double*> b = a; 
    
    // b(0) = 5.0; // ERROR: Compile-time error! b is read-only.
    
    // ---------------------------------------------------------
    // SCENARIO 2: Unmanaged (Weak Reference)
    // We want a View that points to A's data but doesn't affect the
    // reference count (doesn't own the memory).
    // The template constructor handles this because "double*" is compatible.
    // ---------------------------------------------------------
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> c = a;

    // "c" and "a" point to the exact same memory address.
    // modifying "c" modifies "a".
    Kokkos::parallel_for("Modify_C", 1, KOKKOS_LAMBDA(const int i) {
        c(0) = 99.9; 
    });

    // Sync to host to print
    auto a_host = Kokkos::create_mirror_view(a);
    Kokkos::deep_copy(a_host, a);

    std::cout << "Value in A[0] is now: " << a_host(0) << std::endl;
    // Output will be 99.9 because 'c' pointed to 'a'.
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    example_casts();
    Kokkos::finalize();
    return 0;
}
