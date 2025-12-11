#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <iomanip>
template <class ViewType>
void print_slice(ViewType v, int z_index, const std::string& name) {
    std::cout << "\n--- " << name << " (Slice Z=" << z_index << ") ---\n";
    for (int i = 0; i < v.extent(0); ++i) {
        std::cout << "Row " << i << ": [ ";
        for (int j = 0; j < v.extent(1); ++j) {
            std::cout << std::fixed << std::setprecision(1) << v(i, j, z_index) << " ";
        }
        std::cout << "]\n";
    }
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    std::cout << "--- Advanced Kokkos: Manual ROI Construction ---\n";
    const int DIM = 4;
    std::vector<double> raw_data(DIM * DIM * DIM);
    int stride_z = 1;
    int stride_y = 4;
    int stride_x = 16;
    for(int x=0; x<DIM; ++x) {
        for(int y=0; y<DIM; ++y) {
            for(int z=0; z<DIM; ++z) {
                int offset = x*stride_x + y*stride_y + z*stride_z;
                raw_data[offset] = x*100 + y*10 + z;
                std::cout<<"OFFSET: "<<offset<<" raw_data[offset] = "<<raw_data[offset]<<" ";
            }
        }
    }
    std::cout<<std::endl;

    std::cout << "Raw Memory initialized (Size " << raw_data.size() << ").\n";
    int roi_dim_x = 2;
    int roi_dim_y = 2;
    int roi_dim_z = 2;
    int start_offset = (1 * stride_x) + (1 * stride_y) + (1 * stride_z); 
    double* roi_ptr = raw_data.data() + start_offset;
    Kokkos::LayoutStride roi_layout;
    
    roi_layout.dimension[0] = roi_dim_x;
    roi_layout.stride[0]    = stride_x; 

    roi_layout.dimension[1] = roi_dim_y;
    roi_layout.stride[1]    = stride_y; 

    roi_layout.dimension[2] = roi_dim_z;
    roi_layout.stride[2]    = stride_z; 
    using UnmanagedView = Kokkos::View<double***, Kokkos::LayoutStride, 
                                       Kokkos::HostSpace, 
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    UnmanagedView window(roi_ptr, roi_layout);
    std::cout << "\nROI View Created. Checking values...\n";

    double val_000 = window(0, 0, 0);
    std::cout << "Window(0,0,0) [Expect 111.0]: " << val_000 << "\n";
    double val_111 = window(1, 1, 1);
    std::cout << "Window(1,1,1) [Expect 222.0]: " << val_111 << "\n";
    print_slice(window, 0, "ROI Window");
    std::cout << "\nModifying Window(0,0,0) to 999.9...\n";
    window(0,0,0) = 999.9;
    std::cout << "Checking Raw Data at index " << start_offset << " (Original 1,1,1)...\n";
    if (raw_data[start_offset] == 999.9) {
        std::cout << "SUCCESS: The raw memory was updated correctly!\n";
    } else {
        std::cout << "FAILURE: Memory mismatch.\n";
    }
     for(int x=0; x<window.extent(0); ++x) {
       for(int y=0; y<window.extent(1); ++y) {
          for(int z=0; z<window.extent(2); ++z) {
                std::cout<<"window("<<x<<","<<y<<","<<z<<  ")="<<window(x,y,z)<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }

    return 0;
}
