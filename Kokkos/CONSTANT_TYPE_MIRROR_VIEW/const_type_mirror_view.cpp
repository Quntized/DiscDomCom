#include <Kokkos_Core.hpp>
#include <type_traits>
#include <iostream>

template <class AnyView>
void analyze_view_types(AnyView v) {
    // 1. Get the Const (Read-Only) Type
    using ConstView = typename AnyView::const_type;
    
    // 2. Get the Host Mirror (CPU) Type
    using MirrorView = typename AnyView::host_mirror_type;

    std::cout << "Original View is on GPU? ";
#ifdef KOKKOS_ENABLE_CUDA
    // Only compile this check if CUDA is actually enabled
    if (std::is_same_v<typename AnyView::memory_space, Kokkos::CudaSpace>) {
        std::cout << "YES\n";
    } else {
        std::cout << "NO (CPU)\n";
    }
#else
    std::cout << "NO (CUDA disabled)\n";
#endif

    std::cout << "Mirror View is on GPU?   ";
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same_v<typename MirrorView::memory_space, Kokkos::CudaSpace>) {
        std::cout << "YES\n";
    } else {
        std::cout << "NO (CPU)\n";
    }
#else
    std::cout << "NO (CUDA disabled)\n";
#endif
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    
    std::cout << "--- Analyzing View Types ---\n";
    // On a GPU machine, this would default to Cuda.
    using MySpace = Kokkos::DefaultExecutionSpace::memory_space;
    
    Kokkos::View<double*, MySpace> data("my_data", 100);
    
    analyze_view_types(data);
    
    return 0;
}
