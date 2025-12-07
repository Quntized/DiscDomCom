#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Understanding impl_map() Logic ---\n";

    // 1. Create a 2D View (3 Rows, 4 Cols)
    // We use LayoutRight (Row-Major) so:
    // Stride 0 (Rows) = 4 (Must skip 4 cols to get to next row)
    // Stride 1 (Cols) = 1 (Neighbors are contiguous)
    using ViewType = Kokkos::View<double**, Kokkos::LayoutRight>;
    ViewType my_view("A", 3, 4);

    // Fill with data
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            my_view(i, j) = i * 10.0 + j; // e.g., (1, 2) -> 12.0
        }
    }

    // =========================================================
    // SIMULATING impl_map()
    // impl_map() combines the Pointer and the Layout.
    // Let's grab them manually.
    // =========================================================

    // A. The Data (from view_wrap(data()))
    double* raw_ptr = my_view.data();

    // B. The Geometry (from layout())
    auto my_layout = my_view.layout();
    size_t stride_row = my_layout.stride[0]; // Should be 4
    size_t stride_col = my_layout.stride[1]; // Should be 1

    std::cout << "Internal Layout Strides:\n";
    std::cout << "  Row Stride: " << stride_row << "\n";
    std::cout << "  Col Stride: " << stride_col << "\n\n";

    // =========================================================
    // PERFORM THE MAPPING MANUALLY
    // We want to access element at (2, 3) -> Value should be 23.0
    // =========================================================
    int target_row = 2;
    int target_col = 3;

    // 1. The "Official" Way (View handles the map)
    double val_official = my_view(target_row, target_col);

    // 2. The "Impl Map" Way (Manual Calculation)
    // Offset = (row * stride_row) + (col * stride_col)
    size_t offset = (target_row * stride_row) + (target_col * stride_col);
    double val_manual = raw_ptr[offset];

    std::cout << "Target Index (2, 3):\n";
    std::cout << "  View Access:   " << val_official << "\n";
    std::cout << "  Manual Map:    " << val_manual << "\n";
    std::cout << "  Calculated Offset: " << offset << "\n";

    if (val_official == val_manual) {
        std::cout << "\nSUCCESS: We successfully replicated the ViewMapping logic!\n";
    } else {
        std::cout << "\nFAILURE: Math mismatch.\n";
    }

    return 0;
}
