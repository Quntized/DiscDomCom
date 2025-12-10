#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

void run_demonstration() {
    // =========================================================================
    // 1. THE BUILDER (Allocation Constructor)
    // =========================================================================
    // We use the constructor that accepts "Execution Properties" and "Layout".
    // 
    // Properties:
    //   - "ThermalMap": A label for profiling tools.
    //   - Kokkos::WithoutInitializing: Skip zeroing out memory (faster).
    //
    // Layout:
    //   - Kokkos::LayoutRight(10, 5): A 10x5 matrix with Row-Major memory.
    
    using layout_type = Kokkos::LayoutRight;
    
    Kokkos::View<double**, layout_type> parent_view(
        Kokkos::view_alloc("ThermalMap", Kokkos::WithoutInitializing), 
        layout_type(10, 5)
    );

    // Since we used 'WithoutInitializing', the memory contains garbage.
    // We must manually initialize it.
    Kokkos::parallel_for("InitParent", 10, KOKKOS_LAMBDA(const int i) {
        for(int j=0; j<5; ++j) {
            parent_view(i, j) = i * 10.0 + j; // e.g., Row 2, Col 3 -> 23.0
        }
    });
    for (int i = 0; i<10; i++){
      for (int j = 0; j<5; j++){
        std::cout<<parent_view(i,j)<<" ";
      }
      std::cout<<std::endl;
    }

    printf("Created Parent View 'ThermalMap' (10x5).\n");
    std::cout<<"raw pointer of parent view: "<<parent_view.data()<<std::endl;

    // =========================================================================
    // 2. THE SLICER (Subview Constructor)
    // =========================================================================
    // We want to isolate exactly Row 3.
    // In practice, we call the helper 'Kokkos::subview', which internally
    // calls the View constructor: View(src, arg0, arg1...)
    
    auto row_slice = Kokkos::subview(parent_view, 3, Kokkos::ALL);
    std::cout<<std::size(row_slice)<<std::endl;
    for(int i=0; i<std::size(row_slice); i++){
      std::cout<<row_slice(i)<<" ";
    }
    std::cout<<std::endl;
    std::cout<<"Data pointer of row slice: "<<row_slice.data()<<std::endl;

    // row_slice is now a 1D vector of length 5.
    // It points to the SAME memory as parent_view.

    printf("Created Slice of Row 3.\n\n");

    // =========================================================================
    // 3. PROVING THEY ARE CONNECTED
    // =========================================================================
    
    // Modify the SLICE
    Kokkos::parallel_for("ModifySlice", 1, KOKKOS_LAMBDA(const int) {
        // Change index 0 of the slice (which is (3,0) of the parent)
        row_slice(0) = 999.9;
    });
    std::cout<<"Now row slice "<<std::endl;
    for(int i=0; i<std::size(row_slice); i++){
      std::cout<<row_slice(i)<<" ";
    }
    std::cout<<"parent_view again"<<std::endl;
    for (int i = 0; i<10; i++){
      for (int j = 0; j<5; j++){
        std::cout<<parent_view(i,j)<<" ";
      }
      std::cout<<std::endl;
    }

    // Wait for GPU/CPU to finish
    Kokkos::fence();

    // Verify by reading the PARENT
    // We need a Host Mirror to print from CPU
    auto parent_host = Kokkos::create_mirror_view(parent_view);
    Kokkos::deep_copy(parent_host, parent_view);
    
    std::cout<<"Data pointer of parent_host: "<<parent_host.data()<<std::endl;
    std::cout << "Checking Parent View at index (3,0)...\n";
    std::cout << "Expected: 999.9\n";
    std::cout << "Actual:   " << parent_host(3, 0) << "\n";
    std::cout<<"Parent_host view : "<<std::endl;
    for (int i = 0; i<10; i++){
      for (int j = 0; j<5; j++){
        std::cout<<parent_host(i,j)<<" ";
      }
      std::cout<<std::endl;
    }

    if (parent_host(3, 0) == 999.9) {
        std::cout << "SUCCESS: The slice and parent share memory!\n";
    } else {
        std::cout << "FAILURE: Memory was not shared.\n";
    }
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    run_demonstration();
    Kokkos::finalize();
    return 0;
}
