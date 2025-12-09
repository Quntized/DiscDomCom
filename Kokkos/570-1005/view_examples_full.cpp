#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "--- Kokkos View feature demo ---\n";

    // Rank-0 scalar view
    {
      Kokkos::View<double> scalar("scalar");
      scalar() = 3.14159; // rank-0 uses operator()
      std::cout << "Rank-0 scalar value: " << scalar() << "\n";
      std::cout << " scalar.is_allocated(): " << scalar.is_allocated() << "\n";
      std::cout << " scalar.span(): " << scalar.span() << "\n";
    }

    // Rank-1: simple vector, initialize on device and copy to host for printing
    {
      const int N = 16;
      Kokkos::View<int*> v1("v1", N);

      Kokkos::parallel_for("init_v1", N, KOKKOS_LAMBDA(int i) { v1(i) = i * 2; });
      Kokkos::fence();

      auto host_v1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v1);
      std::cout << "Rank-1 v1 extents: " << v1.extent_int(0) << " stride(0): " << v1.stride(0)
                << " span(): " << v1.span() << "\n";
      std::cout << "v1 contents: ";
      for (int i = 0; i < N; ++i) std::cout << host_v1(i) << " ";
      std::cout << "\n";

      // operator[] for rank-1
      std::cout << "v1[3] (via operator[]): " << v1[3] << "\n";
    }

    // Rank-2: LayoutLeft and LayoutRight differences
    {
      const int R = 3, C = 4;
      Kokkos::View<double**, Kokkos::LayoutLeft> left("left", R, C);
      Kokkos::View<double**, Kokkos::LayoutRight> right("right", R, C);

      // initialize both with a parallel_for over an MDRange
      Kokkos::parallel_for("init_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{R,C}), KOKKOS_LAMBDA(int i, int j) {
        left(i,j) = (i+1)*10 + j;
        right(i,j) = (i+1)*100 + j;
      });
      Kokkos::fence();

      auto host_left = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), left);
      auto host_right = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), right);

      std::cout << "LayoutLeft extents: (" << left.extent_int(0) << ", " << left.extent_int(1) << ")"
                << " stride(0): " << left.stride(0) << " stride(1): " << left.stride(1)
                << " span(): " << left.span() << "\n";

      std::cout << "LayoutRight extents: (" << right.extent_int(0) << ", " << right.extent_int(1) << ")"
                << " stride(0): " << right.stride(0) << " stride(1): " << right.stride(1)
                << " span(): " << right.span() << "\n";

      std::cout << "left contents:\n";
      for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) std::cout << host_left(i,j) << " ";
        std::cout << "\n";
      }

      std::cout << "right contents:\n";
      for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) std::cout << host_right(i,j) << " ";
        std::cout << "\n";
      }
    }

    // Rank-3: demonstrate access(...) overloads and extra zero arguments
    {
      const int D0 = 2, D1 = 3, D2 = 4;
      Kokkos::View<int***> v3("v3", D0, D1, D2);

      Kokkos::parallel_for("init_v3", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{D0,D1,D2}), KOKKOS_LAMBDA(int i, int j, int k) {
        v3(i,j,k) = i*100 + j*10 + k;
      });
      Kokkos::fence();

      auto host_v3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v3);
      std::cout << "v3.sample values (operator()): " << host_v3(1,2,3) << "\n";
      // access(...) matches rank and tolerates up to 8 args with trailing zeros
      std::cout << "v3.access(1,2,3): " << v3.access(1,2,3) << "\n";
      std::cout << "v3.access(1,2,3,0,0) (extra zeros): " << v3.access(1,2,3,0,0) << "\n";
      std::cout<<"v3(1,2,3) = "<<v3(1,2,3)<<"\n"<<std::endl;
      //std::cout<<"v3(1,2,3) = "<<v3[1,2,3]<<std::endl; error
    }

    // Unmanaged view from raw pointer and string-label constructor example
    {
      const int N = 5;
      int* raw = static_cast<int*>(std::malloc(sizeof(int)*N));
      for (int i = 0; i < N; ++i) raw[i] = -i;

      Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> raw_view(raw, N);
      std::cout << "unmanaged raw_view span(): " << raw_view.span() << " data(): " << static_cast<const void*>(raw_view.data()) << "\n";
      std::cout << "raw_view contents: ";
      for (int i = 0; i < N; ++i) std::cout << raw_view(i) << " ";
      std::cout << "\n";
      std::free(raw);
    }

    // Subview constructor demo (use View(subview) ctor)
    {
      Kokkos::View<double**> big("big", 4, 6);
      Kokkos::parallel_for("init_big", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{4,6}), KOKKOS_LAMBDA(int i, int j) {
        big(i,j) = i*10 + j;
      });
      Kokkos::fence();

      // subview: take row 2
      auto row2 = Kokkos::subview(big, 2, Kokkos::ALL());
      auto host_row2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), row2);
      for (int i =0; i<big.extent_int(0); i++){
        for(int j =0; j<big.extent_int(1); j++){
          std::cout<<" "<<big(i,j);
        }
        std::cout<<std::endl;
      }
      std::cout << "\nsubview row2 contents: ";
      for (int j = 0; j < row2.extent_int(0); ++j) std::cout << host_row2(j) << " ";
      std::cout << "\n";
    }

    // required_allocation_size demonstration (static query)
    {
      // For a 2D layout, query required allocation size for extents (3,4)
      size_t req = Kokkos::View<int**>::required_allocation_size(3,4);
      std::cout << "required_allocation_size for (3,4): " << req << " bytes (in elements)\n";
    }

    // impl_track() demo (shows allocation tracker presence)
    {
      Kokkos::View<int*> v("vtmp", 8);
      const auto & tracker = v.impl_track();
      std::cout << "impl_track().has_record(): " << (tracker.has_record()? 1 : 0) << "\n";
    }

    std::cout << "--- end demo ---\n";
  }
  Kokkos::finalize();
  return 0;
}
