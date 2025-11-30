// SPDX-License-Identifier: MIT
// Complete example of ddc::UniformPointSampling

#include <ddc/ddc.hpp>
#include <iostream>

// Step 1: Define continuous dimension
struct X {
    static bool constexpr PERIODIC = false;
};

// Step 2: Define discrete dimension (grid)
struct GridX : ddc::UniformPointSampling<X> {};

// Type aliases
using CoordX = ddc::Coordinate<X>;
using IdxX = ddc::DiscreteElement<GridX>;
using IdxStepX = ddc::DiscreteVector<GridX>;
using IdxRangeX = ddc::DiscreteDomain<GridX>;

int main() {
    std::cout << "=== UniformPointSampling Examples ===\n\n";
    
    // -------------------------------------------------------------------------
    // Example 1: Basic init() - Simple uniform grid
    // -------------------------------------------------------------------------
    std::cout << "--- Example 1: Basic Grid ---\n";
    
    // Create grid from x=0.0 to x=10.0 with 11 points
    auto [grid_x, domain_x] = GridX::init<GridX>(
        CoordX(0.0),      // starting point
        CoordX(10.0),     // ending point
        IdxStepX(11)      // number of points
    );
    
    std::cout << "Grid from 0.0 to 10.0 with 11 points\n";
    std::cout << "Domain size: " << domain_x.size() << " points\n";
    
    // Access grid points
    std::cout << "Grid points:\n";
    for (IdxX idx : domain_x) {
        CoordX coord = ddc::coordinate(idx);
        std::cout << "  Index " << (idx - domain_x.front()).value() 
                  << " -> x = " << coord << "\n";
    }
    
    // Calculate spacing
    IdxX first_idx = domain_x.front();
    IdxX second_idx = first_idx + 1;
    double spacing = ddc::coordinate(second_idx) - ddc::coordinate(first_idx);
    std::cout << "Grid spacing: " << spacing << "\n";
    
    // -------------------------------------------------------------------------
    // Example 2: init_ghosted() - Grid with ghost points
    // -------------------------------------------------------------------------
    std::cout << "\n--- Example 2: Grid with Ghost Points ---\n";
    
    // Create grid with 2 ghost points on each side
    auto [grid_ghost, domain_before, domain_main, domain_after, domain_full] = 
        GridX::init_ghosted<GridX>(
            CoordX(0.0),      // a: start of main domain
            CoordX(4.0),      // b: end of main domain
            IdxStepX(5),      // n: 5 points in main domain (0,1,2,3,4)
            IdxStepX(2)       // 2 ghost points on each side
        );
    
    std::cout << "Main domain: [0.0, 4.0] with 5 points\n";
    std::cout << "Ghost points: 2 before, 2 after\n\n";
    
    std::cout << "Domain sizes:\n";
    std::cout << "  Before ghosts: " << domain_before.size() << " points\n";
    std::cout << "  Main domain:   " << domain_main.size() << " points\n";
    std::cout << "  After ghosts:  " << domain_after.size() << " points\n";
    std::cout << "  Full domain:   " << domain_full.size() << " points\n\n";
    
    std::cout << "Ghost points BEFORE main domain:\n";
    for (IdxX idx : domain_before) {
        CoordX coord = ddc::coordinate(idx);
        std::cout << "  Ghost -> x = " << coord << "\n";
    }
    
    std::cout << "Main domain points:\n";
    for (IdxX idx : domain_main) {
        CoordX coord = ddc::coordinate(idx);
        std::cout << "  Main  -> x = " << coord << "\n";
    }
    
    std::cout << "Ghost points AFTER main domain:\n";
    for (IdxX idx : domain_after) {
        CoordX coord = ddc::coordinate(idx);
        std::cout << "  Ghost -> x = " << coord << "\n";
    }
    
    // -------------------------------------------------------------------------
    // Example 3: Fine vs Coarse grids
    // -------------------------------------------------------------------------
    std::cout << "\n--- Example 3: Fine vs Coarse Grids ---\n";
    
    // Coarse grid: 6 points
    auto [grid_coarse, domain_coarse] = GridX::init<GridX>(
        CoordX(0.0), CoordX(5.0), IdxStepX(6)
    );
    
    std::cout << "Coarse grid (6 points):\n";
    for (IdxX idx : domain_coarse) {
        std::cout << "  x = " << ddc::coordinate(idx) << "\n";
    }
    
    // Fine grid: 11 points  
    auto [grid_fine, domain_fine] = GridX::init<GridX>(
        CoordX(0.0), CoordX(5.0), IdxStepX(11)
    );
    
    std::cout << "\nFine grid (11 points):\n";
    for (IdxX idx : domain_fine) {
        std::cout << "  x = " << ddc::coordinate(idx) << "\n";
    }
    
    std::cout << "\n=== Done! ===\n";
    
    return 0;
}
