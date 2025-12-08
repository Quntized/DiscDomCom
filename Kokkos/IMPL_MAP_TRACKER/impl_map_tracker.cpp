#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>

void print_separator(const std::string& title) {
    std::cout << "\n========================================\n" 
              << " " << title << "\n" 
              << "========================================\n";
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    print_separator("Kokkos Internals: impl_map & impl_track");

    // ---------------------------------------------------------
    // PART 1: LIFECYCLE MANAGEMENT (impl_track)
    // ---------------------------------------------------------
    std::cout << "[PART 1] Inspecting Reference Counts via impl_track()\n";

    // 1. Create a Managed View (Owner)
    Kokkos::View<double*> parent("parent", 100);
    
    // Check Tracker: Should have 1 reference (itself)
    // Note: impl_track() returns the internal tracker object.
    // .use_count() is a standard function to check how many people hold the pointer.
    std::cout << "Created 'parent'.\n";
    std::cout << "  Parent Ref Count: " << parent.impl_track().use_count() << " (Expected 1)\n";

    {
        // 2. Create a Copy (Shared Owner)
        Kokkos::View<double*> child = parent;
        std::cout << "Created 'child' from 'parent'.\n";
        
        // Both point to the same tracker
        std::cout << "  Parent Ref Count: " << parent.impl_track().use_count() << " (Expected 2)\n";
        std::cout << "  Child  Ref Count: " << child.impl_track().use_count()  << " (Expected 2)\n";
    } // 'child' dies here

    // 3. Verify Child Death
    std::cout << "'child' went out of scope.\n";
    std::cout << "  Parent Ref Count: " << parent.impl_track().use_count() << " (Expected 1)\n";


    // ---------------------------------------------------------
    // PART 2: UNMANAGED VIEWS (No Tracker)
    // ---------------------------------------------------------
    std::cout << "\n[PART 2] Unmanaged Views\n";

    // Create a view that looks at 'parent' data but doesn't own it.
    // Traits::Unmanaged means "don't track memory".
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ghost = parent;

    // Check Tracker logic from the snippet you pasted:
    // "if (is_unmanaged) return empty_tracker;"
    auto ghost_tracker = ghost.impl_track();

    // An empty/null tracker usually has use_count == 0 (or is invalid)
    std::cout << "Created 'ghost' (Unmanaged) from 'parent'.\n";
    std::cout << "  Ghost Ref Count:  " << ghost_tracker.use_count() << " (Expected 0)\n";
    std::cout << "  Parent Ref Count: " << parent.impl_track().use_count() << " (Still 1 - Ghost doesn't count)\n";

// ---------------------------------------------------------
    // PART 3: THE ENGINE (Manual Calculation)
    // ---------------------------------------------------------
    std::cout << "\n========================================\n" 
              << " PART 3: Manually Driving the Engine \n" 
              << "========================================\n";

    // Let's create a 2D LayoutRight (Row-Major) View.
    // 3 Rows, 4 Columns
    Kokkos::View<double**, Kokkos::LayoutRight> grid("grid", 3, 4);

    // 1. Get the Dimensions and Strides
    // This is the "Blueprint" the engine uses.
    size_t dim0 = grid.extent(0); // 3
    size_t dim1 = grid.extent(1); // 4
    size_t stride0 = grid.stride(0);
    size_t stride1 = grid.stride(1);

    std::cout << "Grid Dimensions: " << dim0 << "x" << dim1 << " (LayoutRight)\n";
    std::cout << "Strides: Row=" << stride0 << ", Col=" << stride1 << "\n";

    // 2. Manually Calculate Offset for Index (1, 1)
    // The engine's formula for LayoutRight is: (i * stride0) + (j * stride1)
    int i = 1;
    int j = 1;
    
    size_t offset_calculated_manually = (i * stride0) + (j * stride1);

    std::cout << "Manually calculated offset for (1, 1): " << offset_calculated_manually << "\n";

    // 3. Verify against the raw pointer
    // Fill the data using the standard API
    grid(1, 1) = 42.0; 

    // Get raw pointer
    double* raw_ptr = grid.data();
    
    // Access memory using our calculated offset
    double value_at_offset = raw_ptr[offset_calculated_manually];

    std::cout << "  Value at View(1,1):     " << grid(1, 1) << "\n";
    std::cout << "  Value at ptr[offset]:   " << value_at_offset << "\n";

    if (grid(1,1) == value_at_offset) {
        std::cout << "SUCCESS: We successfully replicated the engine's logic!\n";
    } else {
        std::cout << "FAILURE: Math mismatch.\n";
    }

    return 0;
}
