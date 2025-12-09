#include <Kokkos_Core.hpp>
#include <iostream>
#include <type_traits>

// Comprehensive Kokkos::View example exercising many APIs shown
// in core/src/Kokkos_View.hpp (stride, span, operator(), operator[],
// impl_map, impl_track, extent_int, access overloads for different ranks).

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos View example\n";

    // -----------------------------
    // Rank-0 (scalar) view
    // -----------------------------
    {
      Kokkos::View<int> scalar("scalar");
      scalar() = 123;
      std::cout << "rank-0 scalar via operator(): " << scalar() << "\n";
      // operator[] would be invalid for rank-0; use operator() to read
      std::cout << "rank-0 scalar via operator() (no operator[] for rank-0): " << scalar() << "\n";
      std::cout << "span() (rank-0) = " << scalar.span() << "\n";
      std::cout << "is_allocated() = " << scalar.is_allocated() << "\n";
    }

    // -----------------------------
    // Rank-1 view
    // -----------------------------
    {
      const int N = 8;
      Kokkos::View<double*> v1("v1", N);

      // fill on host and copy to device (works for host and device builds)
      auto h_v1 = Kokkos::create_mirror_view(v1);
      for (int i = 0; i < N; ++i) h_v1(i) = (double)i * 1.5;
      Kokkos::deep_copy(v1, h_v1);

      std::cout << "\nRank-1 view contents (using operator() and operator[])\n";
      for (int i = 0; i < N; ++i) {
        // both operator() and operator[] are supported by View
        std::cout << "v1(" << i << ") = " << v1(i) << ", v1[" << i << "] = " << v1[i] << "\n";
      }

      // stride for rank-1
      std::cout << "stride(0) = " << v1.stride(0) << "\n";

      // stride(array) writes per-dimension strides and span into s[rank()]
      int s_arr[2] = {0, 0};
      v1.stride(s_arr);
      std::cout << "stride array: s[0] = " << s_arr[0] << ", s[1] (span) = " << s_arr[1] << "\n";

      std::cout << "span() = " << v1.span() << ", span_is_contiguous() = " << v1.span_is_contiguous() << "\n";
      std::cout << "extent_int(0) = " << v1.extent_int(0) << "\n";
    }

    // -----------------------------
    // Rank-2 views with LayoutLeft and LayoutRight
    // -----------------------------
    {
      const int I = 3, J = 4;

      Kokkos::View<int**, Kokkos::LayoutLeft> left("left", I, J);
      Kokkos::View<int**, Kokkos::LayoutRight> right("right", I, J);

      // fill left layout and right layout with distinct values
      for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
          left(i, j) = 100 + i * 10 + j;
          right(i, j) = 200 + i * 10 + j;
        }
      }

      std::cout << "\nRank-2 left layout content (row-major by left semantics)\n";
      for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) std::cout << left(i, j) << " ";
        std::cout << "\n";
      }

      std::cout << "Rank-2 right layout content\n";
      for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) std::cout << right(i, j) << " ";
        std::cout << "\n";
      }

      // Query strides and extents
      std::cout << "left.stride(0) = " << left.stride(0) << ", left.stride(1) = " << left.stride(1) << "\n";
      std::cout << "right.stride(0) = " << right.stride(0) << ", right.stride(1) = " << right.stride(1) << "\n";

      // Use stride array form
      int s2[3] = {0, 0, 0};
      left.stride(s2);
      std::cout << "left stride array: [" << s2[0] << ", " << s2[1] << "] span=" << s2[2] << "\n";

      // Query span via the public API (mapping internals vary across builds)
      std::cout << "left span() = " << left.span() << "\n";
      std::cout << "right span() = " << right.span() << "\n";

      std::cout << "left.span_is_contiguous() = " << left.span_is_contiguous() << "\n";
      std::cout << "right.span_is_contiguous() = " << right.span_is_contiguous() << "\n";
    }

    // -----------------------------
    // Demonstrate access() convenience functions for rank-specific overloads
    // -----------------------------
    {
      Kokkos::View<int***> v3("v3", 2, 3, 4);
      // Fill using nested loops and call access(...) which delegates to base operator()
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
          for (int k = 0; k < 4; ++k) v3(i, j, k) = i * 100 + j * 10 + k;

      std::cout << "\nRank-3 access via access(...) overloads (samples):\n";
      std::cout << "v3.access(1,2,3) = " << v3.access(1, 2, 3) << " (should equal v3(1,2,3))\n";
    }

    // -----------------------------
    // Managed vs unmanaged (note: creating unmanaged views depends on raw pointers)
    // Here we demonstrate wrapping a raw host pointer using view_wrap to construct
    // an unmanaged view that does not own its memory.
    // -----------------------------
    {
      const int M = 5;
      double* raw = new double[M];
      for (int i = 0; i < M; ++i) raw[i] = 3.14 * (i + 1);

      // Construct an unmanaged View using MemoryTraits<Unmanaged>
      auto uview = Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(raw, M);
      std::cout << "\nUnmanaged view data(): " << static_cast<void*>(uview.data()) << ", is_allocated() = " << uview.is_allocated() << "\n";
      std::cout << "Unmanaged elements: ";
      for (int i = 0; i < M; ++i) std::cout << uview(i) << " ";
      std::cout << "\n";
      delete[] raw;
    }

    // -----------------------------
    // Note: impl_track() returns the SharedAllocationTracker reference. For unmanaged
    // views this returns a static empty tracker; for managed views it returns real tracker.
    // We simply demonstrate that calling impl_track() is valid and does not crash.
    // -----------------------------
    {
      Kokkos::View<int*> v("v_tmp", 4);
      // impl_track() exists to allow checking allocation lifecycle from specializations
      (void)v.impl_track();
      std::cout << "Called impl_track() on a managed view (no printed output)." << "\n";
    }

  }
  Kokkos::finalize();
  return 0;
}
