#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    // Create a 2D View (10 rows, 5 columns)
    // LayoutRight (Row-Major):
    // - Stride 1 (Cols) = 1
    // - Stride 0 (Rows) = 5
    Kokkos::View<double**, Kokkos::LayoutRight> my_view("A", 10, 5);

    // Prepare an array to hold the results.
    // Rank is 2, so we need size 3.
    size_t strides[3];

    // CALL THE FUNCTION
    my_view.stride(strides);

    std::cout << "--- Bulk Stride Output ---\n";
    std::cout << "Stride[0] (Row): " << strides[0] << " (Expected 5)\n";
    std::cout << "Stride[1] (Col): " << strides[1] << " (Expected 1)\n";
    
    std::cout << "--- The Magic Value ---\n";
    std::cout << "Stride[2] (Span): " << strides[2] << "\n";
    
    // Explanation of the magic value:
    // Max stride was 5 (Dimension 0). Extent of Dim 0 is 10.
    // Result = 5 * 10 = 50.
    // This means the array spans 50 elements in memory.

    return 0;
}
