#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>
#include <typeinfo>

// A helper to print the simplified name of a type
// (Note: C++ mangled names are messy, we just check if they are the same)
template <typename T>
void print_type_id() {
    std::cout << "  Type Hash: " << typeid(T).hash_code() << "\n";
}

template <class AnyView>
void inspect_view(AnyView v, std::string name) {
    std::cout << "\nInspecting: " << name << "\n";
    
    // 1. The Original Type
    // This is unique for every input.
    std::cout << "  [Original] "; 
    print_type_id<AnyView>();

    // 2. The Uniform Type
    // This converts specific static sizes to generic dynamic pointers.
    using UniformView = typename AnyView::uniform_runtime_type;
    
    std::cout << "  [Uniform ] ";
    print_type_id<UniformView>();
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Uniform Type Demo ---\n";

    // CASE 1: Dynamic 1D Array (double*)
    // Size is set at runtime (10).
    Kokkos::View<double*> dynamic_view("dynamic", 10);

    // CASE 2: Static 1D Array (double[10])
    // Size is fixed at compile time. This is a DIFFERENT C++ type!
    Kokkos::View<double[10]> static_view("static");

    // CASE 3: Another Static Array (double[50])
    // This is YET ANOTHER different C++ type!
    Kokkos::View<double[50]> big_static_view("big_static");

    // Inspect them
    inspect_view(dynamic_view, "Dynamic View (size 10)");
    inspect_view(static_view, "Static View (size 10)");
    inspect_view(big_static_view, "Static View (size 50)");

    std::cout << "\nConclusion:\n";
    std::cout << "Notice that [Original] hashes are all DIFFERENT.\n";
    std::cout << "But [Uniform] hashes are all THE SAME.\n";
    std::cout << "This allows the compiler to merge logic for all three!\n";

    return 0;
}
