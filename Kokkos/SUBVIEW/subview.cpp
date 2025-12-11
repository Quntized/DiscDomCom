#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Specialized Constructors Demo ---\n";
    {
        Kokkos::View<double**> parent("parent", 3, 4);

        Kokkos::parallel_for(3, KOKKOS_LAMBDA(int i) {
            for(int j=0; j<4; ++j) parent(i, j) = i * 10 + j;
        });

        auto slice = Kokkos::subview(parent, 1, Kokkos::ALL);

        std::cout << "\n[1. Subview Constructor]\n";
        std::cout << "Created a slice of 'parent'.\n";
        std::cout << "Parent Extents: " << parent.extent(0) << "x" << parent.extent(1) << "\n";
        std::cout << "Slice Extents:  " << slice.extent(0) << " (Rank-1 Vector)\n";
    }


    {
        std::cout << "\n[2. Allocation Constructor]\n";

        Kokkos::LayoutRight my_layout(5, 5);

        auto props = Kokkos::view_alloc("optimized_view", Kokkos::WithoutInitializing);

        Kokkos::View<double**, Kokkos::LayoutRight> v_opt(props, my_layout);

        std::cout << "Allocated 'v_opt' with custom properties.\n";
        std::cout << "  Label: " << v_opt.label() << "\n";
        std::cout << "  Is Allocated? " << (v_opt.is_allocated() ? "Yes" : "No") << "Memory address"<<v_opt.data()<<"\n";
        v_opt(0, 0) = 99.9;
        std::cout << "  Wrote value 99.9 to (0,0).\n";
    }

    return 0;
}
