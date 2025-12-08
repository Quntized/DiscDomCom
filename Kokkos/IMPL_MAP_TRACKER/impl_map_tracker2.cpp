#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    // 1. Create a View
    Kokkos::View<double**, Kokkos::LayoutRight> grid("grid", 3, 4);
    
    // Fill with data
    for(int i=0; i<3; ++i)
        for(int j=0; j<4; ++j)
            grid(i,j) = i * 10 + j; // 00, 01, ... 10, 11...

    // 2. Extract the Components
    // impl_map() returns the "ViewMapping" object (The Engine)
    auto engine = grid.impl_map();
    
    // data() returns the Raw Pointer (The Fuel)
    double* raw_ptr = grid.data();

    std::cout << "--- impl_map() Demo ---\n";
    std::cout << "Value at (1, 2) using standard View API: " << grid(1, 2) << "\n";

    // 3. Use the Engine Manually
    // The engine's operator() requires the pointer AND the indices.
    // It calculates the offset and returns a reference to the double.
    // Syntax: engine( pointer, index_i, index_j )
    double& val_from_engine = engine(raw_ptr, 1, 2);

    std::cout << "Value at (1, 2) using Engine directly:   " << val_from_engine << "\n";

    // Prove it is a reference
    val_from_engine = 999.0;
    std::cout << "Modified Engine value. View sees:        " << grid(1, 2) << "\n";

    return 0;
}
