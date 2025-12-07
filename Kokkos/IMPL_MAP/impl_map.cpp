#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Internal Mapping Demo ---\n";

    // 1. Create the Original View (The "Parent")
    // 10x10 Matrix on CPU
    Kokkos::View<double**, Kokkos::LayoutRight> parent("parent", 10, 10);

    // Fill with data: value = row index
    for(int i=0; i<10; ++i) {
        for(int j=0; j<10; ++j) {
            parent(i,j) = (double)i;
        }
    }

    // =========================================================
    // WHAT impl_map() DOES (Simulated)
    // =========================================================
    // impl_map() grabs two things: The Data Pointer and The Layout Object.
    
    // A. Grab the Data Pointer (Raw Memory)
    double* raw_ptr = parent.data();

    // B. Grab the Layout (The "Map")
    // This object contains the Dimensions (10, 10) and Strides.
    auto layout_info = parent.layout();

    std::cout << "Original Layout: " 
              << layout_info.dimension[0] << "x" << layout_info.dimension[1] <<"another info , if exist: "<<layout_info.dimension[2] <<"another info , if exist: "<<layout_info.dimension[3] <<"\n";

    // =========================================================
    // RECONSTRUCTION
    // =========================================================
    // We can now create a NEW View (a "Shadow") using exactly what impl_map() returns.
    // We use "Unmanaged" because we don't want this new view to delete the memory
    // when it dies (the parent still owns it).
    
    using UnmanagedViewType = Kokkos::View<double**, Kokkos::LayoutRight, 
                                           Kokkos::HostSpace, 
                                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Constructor taking (Pointer, Layout) -> This internally calls the mapping logic!
    UnmanagedViewType shadow_view(raw_ptr, layout_info);

    // =========================================================
    // VERIFICATION
    // =========================================================
    std::cout << "\nVerifying Shadow View...\n";
    
    // Modify the SHADOW view
    shadow_view(5, 5) = 999.99;

    // Check the PARENT view
    std::cout << "Modified Shadow(5,5). Checking Parent(5,5): " 
              << parent(5,5) << " (Expected 999.99)\n";

    if (parent(5,5) == 999.99) {
        std::cout << "SUCCESS: The mapping was reconstructed perfectly.\n";
    } else {
        std::cout << "FAILURE: The views are disconnected.\n";
    }

    return 0;
}
