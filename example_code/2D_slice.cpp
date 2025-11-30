#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp> // Required for ddc::for_each

struct Space;
struct Time;

// Define Grids
struct GridSpace : ddc::UniformPointSampling<Space> {};
struct GridTime : ddc::UniformPointSampling<Time> {};

using IdxSpace = ddc::DiscreteElement<GridSpace>;
using IdxTime = ddc::DiscreteElement<GridTime>;
using IdxRangeST = ddc::DiscreteDomain<GridSpace, GridTime>;

int main(int argc, char** argv) {
    // Initialize Kokkos (Always needed for DDC)
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    // --- FIX 1: EXPLICIT TYPES ---
    // You cannot pass '0.0' or '5'. You must wrap them.
    auto [impl_s, dom_s] = GridSpace::init<GridSpace>(
        ddc::Coordinate<Space>(0.0), 
        ddc::Coordinate<Space>(10.0), 
        ddc::DiscreteVector<GridSpace>(5)
    );

    auto [impl_t, dom_t] = GridTime::init<GridTime>(
        ddc::Coordinate<Time>(0.0), 
        ddc::Coordinate<Time>(1.0), 
        ddc::DiscreteVector<GridTime>(3)
    );
    
    // Combine into 2D domain
    IdxRangeST domain_2d(dom_s, dom_t);

    // Create 2D Chunk
    ddc::Chunk<double, IdxRangeST> simulation_data(domain_2d);

    // --- FIX 2: USE ddc::for_each FOR 2D LOOPS ---
    // Standard 'for(auto ix : domain_2d)' does not work for multidimensional grids.
    ddc::for_each(domain_2d, [&](ddc::DiscreteElement<GridSpace, GridTime> ix) {
        simulation_data(ix) = 42.0; 
    });

    // 3. SLICING!
    // We want to access the "Time = 1" row.
    // We create an index for Time.
    IdxTime time_idx = dom_t.front() + 1;

    // This calls the operator[] we discussed
    // It returns a 1D ChunkSpan (Space only).
    auto time_slice = simulation_data[time_idx];

    std::cout << "Original is 2D. Slice is 1D.\n";
    std::cout << "Slice size: " << time_slice.size() << " (Expected 5)\n";

    // Modifying the slice...
    // Note: time_slice is 1D, so we access it with dom_s.front() (a Space index)
    time_slice(dom_s.front()) = 99.9;

    // ...actually modifies the original Chunk!
    std::cout << "Original data at (0, 1) is now: " 
              << simulation_data(dom_s.front(), time_idx) << "\n";

    return 0;
}
