#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Span vs Size Demo ---\n";

    // 1. Create a 2D Parent View (10 rows, 10 cols)
    // We explicitly use LayoutRight (Standard C++ Row-Major).
    // In memory, Row 0 is contiguous, then Row 1, etc.
    using ViewType = Kokkos::View<double**, Kokkos::LayoutRight>;
    ViewType parent("parent", 10, 10);
    
    // Fill it with recognizable data
    // Value = Row.Col (e.g., 5.3 is Row 5, Col 3)
    for(int i=0; i<10; ++i) {
        for(int j=0; j<10; ++j) {
            parent(i, j) = i + (j * 0.1);
            std::cout<<"Parent("<<i<<", "<<j<<" ) = "<<parent(i,j)<<std::endl;
        }
    }

    std::cout << "\n[Parent View (10x10)]\n";
    std::cout << "  size():       " << parent.size() << "\n";
    std::cout << "  span():       " << parent.span() << "\n";
    std::cout << "  Is Contiguous? " << (parent.span_is_contiguous() ? "Yes" : "No") << "\n";

    // 2. Create a STRIDED Slice (Take Column 0)
    // We want all rows (Kokkos::ALL) for column index 0.
    // Because data is stored Row-by-Row, jumping down a column requires
    // skipping over entire rows. This creates a large 'stride'.
    auto col_slice = Kokkos::subview(parent, Kokkos::ALL, 0);


    std::cout << "\n[Column Slice (Column 0)]\n";
    std::cout << "  size():       " << col_slice.size() << " (10 elements)\n";
    
    // SPAN EXPLANATION:
    // To go from parent(0,0) to parent(9,0), we have to span almost the 
    // entire memory block of the 10x10 array.
    // Span should be roughly 91 (Address of last element - Address of first).
    std::cout << "  span():       " << col_slice.span() << " (Notice > size!)\n";
    std::cout << "  Is Contiguous? " << (col_slice.span_is_contiguous() ? "Yes" : "No") << "\n";
    
    std::cout << "\nVerifying Slice Data:\n";
    std::cout << col_slice.extent(0)<<std::endl;
    std::cout << col_slice.extent(1)<<std::endl;;
    for(int i=0; i<col_slice.extent(0); ++i) {
        std::cout << "  Slice(" << i << ") = " << col_slice(i) << "\n";
    }

    return 0;
}
