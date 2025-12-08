#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Rank-1 Offset Logic Demo ---\n";

    // =========================================================
    // CASE 1: Standard View (Contiguous)
    // "Fast Path": compute_offset simply returns 'index'.
    // =========================================================
    Kokkos::View<double*> standard("standard", 10);
    
    // Fill with data: 10.0, 11.0, 12.0 ...
    for(int i=0; i<10; ++i) standard(i) = 10.0 + i;

    // Let's look at Index 3
    int idx = 3;
    
    // The "Fast Path" Math:
    size_t offset_fast = idx; // Just the index itself!

    std::cout << "\n[Case 1: Standard View]\n";
    std::cout << "  Target Index: " << idx << "\n";
    std::cout << "  Stride(0):    " << standard.stride(0) << " (1 means contiguous)\n";
    std::cout << "  Raw Pointer:  " << standard.data() << "\n";
    
    // Verify Access via raw pointer
    double val_fast = standard.data()[offset_fast];
    std::cout << "  Value at offset " << offset_fast << ": " << val_fast << "\n";


    // =========================================================
    // CASE 2: Strided Slice (Non-Contiguous)
    // "Slow Path": compute_offset returns 'index * stride'.
    // =========================================================
    
    // We want to simulate a view that steps by 2.
    // Logical indices: 0, 1, 2, 3, 4
    // Maps to Parent:  0, 2, 4, 6, 8
    
    // 1. Define the Layout
    // Dimension 0 has size 5.
    // Stride 0 is 2.
    Kokkos::LayoutStride stride_layout;
    stride_layout.dimension[0] = 5;
    stride_layout.stride[0] = 2;

    // 2. Create the View
    // We use Unmanaged because we are borrowing 'standard's memory.
    // We pass the raw pointer and the specific layout.
    Kokkos::View<double*, Kokkos::LayoutStride, Kokkos::HostSpace, 
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> slice(standard.data(), stride_layout);

    // Let's look at Index 3 of the SLICE
    // This corresponds to the 4th element of the slice.
    // The "Slow Path" Math logic inside Kokkos is: index * stride
    
    size_t stride = slice.stride(0); // Should be 2
    size_t offset_slow = idx * stride; 

    std::cout << "\n[Case 2: Strided Slice]\n";
    std::cout << "  Target Index: " << idx << " (Logical index inside slice)\n";
    std::cout << "  Stride(0):    " << stride << "\n";
    std::cout << "  Raw Pointer:  " << slice.data() << " (Same backing memory)\n";
    std::cout << "  Calculated Offset: " << idx << " * " << stride << " = " << offset_slow << "\n";

    // Verify Access
    // Note: slice.data() points to the start of 'standard', so we use that pointer.
    // We verify that offset 6 holds the value 16.0 (10.0 + 6)
    double val_slow = slice.data()[offset_slow];
    
    std::cout << "  Value at offset " << offset_slow << ": " << val_slow << "\n";
    std::cout << "  Value via View API:  " << slice(idx) << "\n";

    if (val_slow == slice(idx)) {
        std::cout << "SUCCESS: Manual offset calculation matches!\n";
    }

    return 0;
}
