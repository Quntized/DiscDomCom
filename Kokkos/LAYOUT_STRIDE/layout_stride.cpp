#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Stride Demo ---\n";

    // 1. LayoutRight (C++ Style)
    // 10 Rows, 5 Columns
    Kokkos::View<double**, Kokkos::LayoutRight> view_right("Right", 10, 5);

    std::cout << "\n[LayoutRight] (10x5)\n";
    std::cout << "  Stride(0) [Row Jump]: " << view_right.stride(0) << " (Expected 5)\n";
    std::cout << "  Stride(1) [Col Jump]: " << view_right.stride(1) << " (Expected 1)\n";

    // 2. LayoutLeft (Fortran/GPU Style)
    // 10 Rows, 5 Columns
    Kokkos::View<double**, Kokkos::LayoutLeft> view_left("Left", 10, 5);

    std::cout << "\n[LayoutLeft] (10x5)\n";
    std::cout << "  Stride(0) [Row Jump]: " << view_left.stride(0) << " (Expected 1)\n";
    std::cout << "  Stride(1) [Col Jump]: " << view_left.stride(1) << " (Expected 10)\n";

    return 0;
}
