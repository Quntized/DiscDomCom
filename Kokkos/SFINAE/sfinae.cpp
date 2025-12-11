#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>


template <typename T>
void print_view_type(const std::string& name) {
    std::cout << "View '" << name << "' traits:\n";
    std::cout << "  Is Const? " 
              << (std::is_const_v<typename T::value_type> ? "YES" : "NO") << "\n";
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Converting Constructor Demo ---\n";


    using MutableType = Kokkos::View<double*>;
    MutableType v_write("writable", 10);
    
    v_write(0) = 42.0;

    std::cout << "\n[Step 1] Created Writable View.\n";
    print_view_type<MutableType>("v_write");
    std::cout << "  Value: " << v_write(0) << "\n";



    using ConstType = Kokkos::View<const double*>;
    
    ConstType v_read = v_write; 
    std::cout << "\n[Step 2] Created Const View (v_read = v_write).\n";
    print_view_type<ConstType>("v_read");

    std::cout << "  Value seen by v_read: " << v_read(0) << "\n";

    using LayoutRightType = Kokkos::View<double**, Kokkos::LayoutRight>;
    LayoutRightType matrix("matrix", 3, 3);
    
    using StridedType = Kokkos::View<double**, Kokkos::LayoutStride>;
    
    StridedType generic_matrix = matrix; 

    std::cout << "\n[Step 3] Layout Conversion.\n";
    std::cout << "  Converted specific LayoutRight to generic LayoutStride successfully.\n";

    return 0;
}
