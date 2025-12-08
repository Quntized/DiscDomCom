#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    // 1. Create a 1D View
    Kokkos::View<double*> view_1d("1D", 10);

    // USAGE OF operator[] (The code you analyzed)
    // Valid for Rank-1 views.
    view_1d[0] = 3.14;
    view_1d[1] = 9.99;

    std::cout << "1D Access via []: " << view_1d[0] << "\n";

    // 2. Create a 2D View
    Kokkos::View<double**> view_2d("2D", 5, 5);

    // USAGE OF operator() (The Multi-D operator)
    // operator[] usually cannot handle (i, j).
    view_2d(0, 0) = 1.0;
    view_2d(0, 1) = 2.0;

    // ERROR DEMONSTRATION (Concept)
    // If you enabled bounds checking and tried to call internal accessors
    // for a 3rd dimension on this 2D view, the Debug Checker would catch it:
    // view_2d.access(0, 0, 5, 0, 0, 0, 0, 0); // <--- '5' is invalid! Crash!

    std::cout << "2D Access via (): " << view_2d(0, 1) << "\n";

    return 0;
}
