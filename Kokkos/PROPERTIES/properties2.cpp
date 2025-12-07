#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>
#include <vector>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Comprehensive API Coverage ---\n";

    // 1. Create a View
    // 10x10 Matrix
    Kokkos::View<double**, Kokkos::LayoutRight> my_view("A", 10, 10);

    // =========================================================
    // API 1: is_allocated()
    // =========================================================
    Kokkos::View<double> empty_view; // Default constructor (no allocation)
    
    std::cout << "\n[1. is_allocated()]\n";
    std::cout << "  Empty View: " << (empty_view.is_allocated() ? "Allocated" : "Unallocated (NULL)") << "\n";
    std::cout << "  Real View:  " << (my_view.is_allocated()    ? "Allocated" : "Unallocated (NULL)") << "\n";

    // =========================================================
    // API 2: reference_type_is_lvalue_reference
    // =========================================================
    // This checks: "When I type view(0), do I get a real variable address or a temporary proxy?"
    std::cout << "\n[2. reference_type_is_lvalue_reference]\n";
    
    // Check our standard view
    if constexpr (decltype(my_view)::reference_type_is_lvalue_reference) {
        std::cout << "  my_view: TRUE. (It returns 'double&', you can take its address &view(0))\n";
    } else {
        std::cout << "  my_view: FALSE. (It returns a Proxy, common in Atomic Views)\n";
    }

    // =========================================================
    // API 3: extent_int(r)
    // =========================================================
    // Standard .extent(r) returns 'size_t' (unsigned long). 
    // .extent_int(r) returns 'int'. Useful for legacy C libraries (BLAS/LAPACK).
    std::cout << "\n[3. extent_int(r)]\n";
    std::cout << "  Dimension 0 (Rows): " << my_view.extent_int(0) << " (Type: int)\n";
    std::cout << "  Dimension 1 (Cols): " << my_view.extent_int(1) << " (Type: int)\n";

    // =========================================================
    // API 4: data()
    // =========================================================
    // Returns the raw pointer.
    std::cout << "\n[4. data()]\n";
    double* raw_ptr = my_view.data();
    std::cout << "  Raw Pointer Address: " << raw_ptr << "\n";
    
    // We can prove it works by writing to the pointer directly
    raw_ptr[0] = 42.0; 
    std::cout << "  Wrote 42.0 via pointer. View sees: " << my_view(0,0) << "\n";

    // =========================================================
    // API 5 & 6: span() and span_is_contiguous()
    // =========================================================
    // To demonstrate this, we need a STRIDED slice (Column 0).
    // In LayoutRight, a column is NOT contiguous in memory.
    auto col_slice = Kokkos::subview(my_view, Kokkos::ALL, 0);

    std::cout << "\n[5 & 6. span() vs size() and is_contiguous()]\n";
    
    std::cout << "  --- Parent View (Contiguous) ---\n";
    std::cout << "  Size (Elements): " << my_view.size() << "\n";
    std::cout << "  Span (Mem Slots):" << my_view.span() << "\n";
    std::cout << "  Contiguous?      " << (my_view.span_is_contiguous() ? "YES" : "NO") << "\n";

    std::cout << "  --- Column Slice (Non-Contiguous) ---\n";
    std::cout << "  Size (Elements): " << col_slice.size() << " (10 elements)\n";
    std::cout << "  Span (Mem Slots):" << col_slice.span() << " (Start to End distance)\n";
    std::cout << "  Contiguous?      " << (col_slice.span_is_contiguous() ? "YES" : "NO") << "\n";

    return 0;
}
