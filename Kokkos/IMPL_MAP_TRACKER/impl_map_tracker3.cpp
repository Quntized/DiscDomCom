#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    // 1. Create a View
    Kokkos::View<double**, Kokkos::LayoutRight> grid("grid", 3, 4);
    
    // Fill with data
    for(int i=0; i<3; ++i)
        for(int j=0; j<4; ++j)
            grid(i,j) = i * 10 + j; 

    // 2. Extract the "Engine Components" (Layout & Pointer)
    // impl_map() internally holds these exact two things.
    auto layout = grid.layout();
    double* raw_ptr = grid.data();

    // 3. Be The Engine (Manual Calculation)
    // The engine's job is to calculate: Offset = (i * stride0) + (j * stride1)
    
    // Get strides from the layout object
    size_t stride0 = grid.stride(0); // Row stride
    size_t stride1 = grid.stride(1); // Col stride (usually 1 for LayoutRight)

    std::cout << "--- Mapping Engine Demo ---\n";
    std::cout << "Strides: Row=" << stride0 << ", Col=" << stride1 << "\n";

    // Let's access index (1, 2) manually
    int target_i = 1;
    int target_j = 2;

    // PERFORM THE MAPPING MATH
    size_t offset = (target_i * stride0) + (target_j * stride1);
    
    // Access the memory directly
    double& val_from_manual_map = raw_ptr[offset];

    std::cout << "Value at (1, 2) using standard View API: " << grid(target_i, target_j) << "\n";
    std::cout << "Value at (1, 2) using Manual Calculation:  " << val_from_manual_map << "\n";

    // Prove it is a reference
    val_from_manual_map = 999.0;
    std::cout << "Modified Raw Memory. View sees:            " << grid(target_i, target_j) << "\n";

    if (grid(target_i, target_j) == 999.0) {
        std::cout << "SUCCESS: We successfully replicated the ViewMapping logic!\n";
    }

    return 0;
}
