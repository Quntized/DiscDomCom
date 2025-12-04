#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    // 1. Create a View with DYNAMIC dimensions (size set at runtime)
    //    Type: double* (pointer)
    Kokkos::View<double*> dynamic_view("dynamic", 100);

    // 2. Create a View with STATIC dimensions (size fixed in type)
    //    Type: double[50] (array)
    //    This is common in small physics kernels.
    Kokkos::View<double[50]> static_view("static");

    // 3. THE ASSIGNMENT CHECK
    
    // Scenario A: Can I assign static -> dynamic?
    // Logic: dynamic_view is dynamic (rank_dynamic >= 1). It accepts anything.
    // Result: TRUE.
    bool can_assign_A = Kokkos::is_assignable(dynamic_view, static_view);
    
    // Scenario B: Can I assign dynamic -> static?
    // Logic: static_view is NOT dynamic. 
    //        So it checks: static_extent(0) == src.extent(0)?
    //        50 == 100? -> FALSE.
    // Result: FALSE.
    bool can_assign_B = Kokkos::is_assignable(static_view, dynamic_view);

    std::cout << "Can assign Static(50) -> Dynamic? " << std::boolalpha << can_assign_A << "\n";
    std::cout << "Can assign Dynamic(100) -> Static(50)? " << can_assign_B << "\n";

    return 0;
}
