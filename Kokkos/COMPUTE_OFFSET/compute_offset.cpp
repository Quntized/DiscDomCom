#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Multi-Dim Offset Logic Demo ---\n";

    // =========================================================
    // 1. Create a 3D View (LayoutRight / Standard C++)
    // Dimensions: 4 Planes (Depth), 3 Rows (Height), 2 Cols (Width)
    // =========================================================
    Kokkos::View<double***, Kokkos::LayoutRight> grid("grid_3d", 4, 3, 2);

    // Fill with unique data to verify later
    // Value = 100*i + 10*j + k
    for(int i=0; i<4; ++i)
        for(int j=0; j<3; ++j)
            for(int k=0; k<2; ++k)
                grid(i,j,k) = (i * 100) + (j * 10) + k;

    // =========================================================
    // 2. Extract Strides (The "Weights")
    // =========================================================
    // For LayoutRight (Row-Major):
    // Stride 2 (Last dim)  = 1 (Contiguous)
    // Stride 1 (Middle)    = Dim2_Size = 2
    // Stride 0 (First dim) = Dim1_Size * Dim2_Size = 3 * 2 = 6
    size_t s0 = grid.stride(0);
    size_t s1 = grid.stride(1);
    size_t s2 = grid.stride(2);

    std::cout << "\n[Layout Geometry]\n";
    std::cout << "  Dimensions: 4 x 3 x 2\n";
    std::cout << "  Strides:    S0=" << s0 << ", S1=" << s1 << ", S2=" << s2 << "\n";

    // =========================================================
    // 3. Manual Offset Calculation
    // The snippet you posted uses a 'Fold Expression' to do this:
    // Offset = (i * s0) + (j * s1) + (k * s2)
    // =========================================================
    
    // Target Index: (1, 2, 1) -> Represents (Plane 1, Row 2, Col 1)
    int i=1, j=2, k=1;

    size_t offset = (i * s0) + (j * s1) + (k * s2);

    // Let's trace the math:
    // i=1 * 6 = 6
    // j=2 * 2 = 4
    // k=1 * 1 = 1
    // Total   = 11
    
    std::cout << "\n[Target Index: (1, 2, 1)]\n";
    std::cout << "  Manual Math: (" << i << "*" << s0 << ") + (" 
                                    << j << "*" << s1 << ") + (" 
                                    << k << "*" << s2 << ")\n";
    std::cout << "  Calculated Offset: " << offset << "\n";

    // =========================================================
    // 4. Verify against Raw Memory
    // =========================================================
    double* raw_ptr = grid.data();
    double val_at_offset = raw_ptr[offset];
    
    std::cout << "  Value in Memory:   " << val_at_offset << "\n";
    std::cout << "  Value via View API:" << grid(i, j, k) << "\n";

    if (val_at_offset == grid(i,j,k)) {
        std::cout << "SUCCESS: The multi-dimensional math works!\n";
    }

    return 0;
}
