#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Bridge Constructor Demo ---\n";
    int N = 20;
    std::vector<double> raw_data(N);
    for(int i=0; i<N; ++i) raw_data[i] = (double)i;
    Kokkos::LayoutStride my_layout;
    my_layout.dimension[0] = 4;
    my_layout.stride[0]    = 5; 
    my_layout.dimension[1] = 2;
    my_layout.stride[1]    = 1;
    using ViewType = Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace, 
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    ViewType v_strided(raw_data.data(), my_layout);

    std::cout << "Wrapped raw pointer with LayoutStride.\n";
    std::cout << "View Dimensions: " << v_strided.extent(0) << "x" << v_strided.extent(1) << "\n";
    std::cout << "v(0, 1) should be 1.0: " << v_strided(0, 1) << "\n";
    std::cout << "v(1, 0) should be 4.0: " << v_strided(1, 0) << "\n";

    if (v_strided(1, 0) == 4.0) {
        std::cout << "SUCCESS: The bridge constructor mapped the strides correctly!\n";
    }
    for (int i =0; i<size(raw_data); i++){
      std::cout<<raw_data[i]<<" ";
    }
    std::cout<<std::endl;
    for (int i=0; i<v_strided.extent(0); i++){
      for(int j =0; j<v_strided.extent(1); j++){
        std::cout<<v_strided(i,j) <<" "; 
      }
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
    return 0;
}
