#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    // 1. Initialize Kokkos
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "--- Kokkos Memory Introspection Demo ---\n";

    // ---------------------------------------------------------
    // PART 1: required_allocation_size
    // ---------------------------------------------------------
    {
        // We want to know: How many bytes does a 3x4 matrix of ints take?
        size_t req_bytes = Kokkos::View<int**>::required_allocation_size(3, 4);
        
        std::cout << "Calculated Size for View<int**>(3, 4):\n";
        std::cout << "  Kokkos Req: " << req_bytes << " bytes\n";
    }

    // ---------------------------------------------------------
    // PART 2: impl_track() usage
    // ---------------------------------------------------------
    {
        std::cout << "\nChecking Allocation Tracker:\n";
        
        // Create a real managed view
        Kokkos::View<int*> v("vtmp", 8);
        
        // Get the internal tracker (Use reference & to avoid copying/locking)
        const auto & tracker = v.impl_track();
        

        bool status = tracker.has_record();
        
        std::cout << "  View 'vtmp' created.\n";
        std::cout << "  tracker.has_record(): " << (status ? "YES (1)" : "NO (0)") << "\n";
        
        // Verify what happens with an empty view
        Kokkos::View<int*> empty_view;
        
        // Check empty view tracker
        std::cout << "  Empty View tracker has record? " 
                  << (empty_view.impl_track().has_record() ? "YES" : "NO") << "\n";
    }

    std::cout << "--- end demo ---\n";

    return 0;
}
