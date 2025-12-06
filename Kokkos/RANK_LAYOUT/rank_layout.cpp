#include <Kokkos_Core.hpp>
#include <iostream>

// Helper to print rank info
template <class ViewType>
void inspect_view(const std::string& name) {
    std::cout << "View: " << name << "\n";
    std::cout << "  Total Rank:   " << ViewType::rank << "\n";
    std::cout << "  Dynamic Rank: " << ViewType::rank_dynamic << "\n";
    std::cout << "---------------------------\n";
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    // Case 1: Fully Dynamic 2D Array
    // Both dimensions defined at runtime.
    using Dynamic2D = Kokkos::View<double**>;
    inspect_view<Dynamic2D>("double**");

    // Case 2: Mixed 2D Array
    // Dim 0 is dynamic (*), Dim 1 is static ([5]).
    using Mixed2D = Kokkos::View<double*[5]>;
    inspect_view<Mixed2D>("double*[5]");

    // Case 3: Fully Static 2D Array
    // Both dimensions fixed at compile time.
    using Static2D = Kokkos::View<double[10][10]>;
    inspect_view<Static2D>("double[10][10]");

    // --- Using layout() ---
    Kokkos::View<double**> my_data("my_data", 10, 20);
    
    // Get the layout object from the instance
    auto my_layout = my_data.layout();

    std::cout << "\nRuntime Layout Check:\n";
    std::cout << "  Dimension 0: " << my_layout.dimension[0] << " (Expected 10)\n";
    std::cout << "  Dimension 1: " << my_layout.dimension[1] << " (Expected 20)\n";

    return 0;
}
