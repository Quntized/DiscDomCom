#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    std::cout << "--- Kokkos Wrapping Constructor Demo ---\n";
    int N = 10;
    std::vector<double> raw_data(N);
    for(int i=0; i<N; ++i) raw_data[i] = i * 1.1;
    using UnmanagedView = Kokkos::View<double*, Kokkos::HostSpace, 
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::LayoutRight layout(N);
    UnmanagedView v_wrapped(raw_data.data(), layout);

    std::cout << "Created View '" << v_wrapped.label() << "' wrapping raw pointer.\n";
    std::cout << "  v_wrapped(0) = " << v_wrapped(0) << "\n";
    std::cout << "  v_wrapped(5) = " << v_wrapped(5) << "\n";
    v_wrapped(0) = 999.9;
    std::cout << "Modified View(0). Raw vector[0] is now: " << raw_data[0] << "\n";
    if (raw_data[0] == 999.9) {
        std::cout << "SUCCESS: The View and Vector share the same memory.\n";
    }
    return 0;
}
