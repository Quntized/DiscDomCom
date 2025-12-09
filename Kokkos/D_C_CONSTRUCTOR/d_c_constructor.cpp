#include <Kokkos_Core.hpp>
#include <iostream>

void print_ref_count(const std::string& name, Kokkos::View<double*> v) {
    // We use the impl_track() function you learned earlier!
    std::cout << "  " << name << " Ref Count: " 
              << v.impl_track().use_count() << "\n";
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    std::cout << "--- Kokkos View Lifecycle Demo ---\n";

    // 1. Default Constructor
    // Corresponds to: View() = default;
    Kokkos::View<double*> a;
    std::cout << "1. Default Constructed 'a'. Is Allocated? " 
              << (a.is_allocated() ? "Yes" : "No") << "\n";

    // 2. Assignment Operator (Constructor with name)
    a = Kokkos::View<double*>("a", 100);
    print_ref_count("a", a);

    // 3. Copy Constructor (Shallow Copy)
    // Corresponds to: View(const View& other)
    {
        std::cout << "\n2. Copy Constructing 'b' from 'a'...\n";
        Kokkos::View<double*> b(a); 
        
        // Both point to same memory. Ref count increments.
        print_ref_count("a", a);
        print_ref_count("b", b);
    } 
    // 'b' is destroyed here (~View called). Ref count decrements.
    
    std::cout << "\n'b' went out of scope.\n";
    print_ref_count("a", a);

    // 4. Move Constructor
    // Corresponds to: View(View&& other)
    std::cout << "\n3. Move Constructing 'c' from temporary...\n";
    
    // We create a temporary view and immediately move it into 'c'.
    Kokkos::View<double*> c(Kokkos::View<double*>("temp", 50));
    
    print_ref_count("c", c);

    return 0;
}
