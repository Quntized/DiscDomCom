#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Legacy Constructor Demo ---\n";
    std::vector<int> raw_data(10);
    for(int i=0; i<10; ++i) raw_data[i] = i;
    Kokkos::LayoutStride legacy_layout;
    legacy_layout.dimension[0] = 4;
    legacy_layout.stride[0] = 3;
    using ViewType = Kokkos::View<int*, Kokkos::LayoutStride, Kokkos::HostSpace, 
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    ViewType v_stride(raw_data.data(), legacy_layout);

    std::cout << "View created from pointer + Legacy LayoutStride object.\n";
    std::cout << "  Internal Layout Type is 'layout_stride'? " 
              << (std::is_same_v<ViewType::array_layout, Kokkos::LayoutStride> ? "Yes" : "No") 
              << "\n";

    std::cout << "  v(0) [Raw 0]: " << v_stride(0) << "\n";
    std::cout << "  v(1) [Raw 2]: " << v_stride(1) << "\n";
    std::cout << "  v(2) [Raw 4]: " << v_stride(2) << "\n";
    for (int i =0; i<size(raw_data); i++){
      std::cout<<raw_data[i]<<" ";
    }
    std::cout<<std::endl;
    for (int i=0; i<v_stride.extent(0); i++){
      std::cout<<v_stride[i] <<" "; 
    }
    std::cout<<std::endl;
    return 0;
}
