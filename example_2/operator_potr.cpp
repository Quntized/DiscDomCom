#include <ddc/ddc.hpp>
#include <iostream>

// --- Setup (Same as your example) ---
struct X { static bool constexpr PERIODIC = false; };
struct GridX : ddc::UniformPointSampling<X> {};
using CoordX = ddc::Coordinate<X>;
using IdxX = ddc::DiscreteElement<GridX>;
using IdxStepX = ddc::DiscreteVector<GridX>;
using IdxRangeX = ddc::DiscreteDomain<GridX>;

int main() {
    // 1. Create a grid
    auto [impl, domain] = GridX::init<GridX>(
        CoordX(100.0), // Start at 100.0
        CoordX(200.0), // End at 200.0
        IdxStepX(11)   // 11 points (indices 0..10)
    );

    // 2. Get an iterator to the beginning of the domain
    // 'it' is the "pointer". It's an object pointing to the first element.
    // Its internal m_value is the DiscreteElement for index 0.
    auto it = domain.begin();

    // -----------------------------------------------------------------
    // THIS IS THE CALL to operator*()
    // -----------------------------------------------------------------
    // You write this:
    IdxX first_index = *it;

    // The compiler translates it to this function call:
    // IdxX first_index = it.operator*();

    // What happens inside the function:
    // 1. The function runs: `return m_value;`
    // 2. The iterator's internal `m_value` (the index 0) is returned.
    // 3. 'first_index' is now a copy of that index.
    // -----------------------------------------------------------------

    std::cout << "The iterator 'it' points to the index: " << first_index << "\n";
std::cout << "The coordinate at this index is: " << impl(ddc::DiscreteCoordinate<GridX>(first_index)) << "\n\n";

    // 4. Move the iterator forward
    it += 3; // Now 'it' points to index 3

    // -----------------------------------------------------------------
    // CALLING operator*() AGAIN
    // -----------------------------------------------------------------
    IdxX fourth_index = *it; // Calls it.operator*() again
    // -----------------------------------------------------------------

    std::cout << "The iterator 'it' now points to the index: " << fourth_index << "\n";
std::cout << "The coordinate at this index is: " << impl(ddc::DiscreteCoordinate<GridX>(fourth_index)) << "\n\n";

    // 5. Common use in a traditional for-loop
    // (A for-each loop hides this, but this is what's happening)
    std::cout << "Iterating with operator*():\n";
    for (auto iter = domain.begin(); iter != domain.end(); ++iter) {
        
        // Use operator*() to get the index at the iterator's position
        IdxX current_index = *iter;
        
        std::cout << "  Index: " << current_index 
                  << " -> Coordinate: " << impl(ddc::DiscreteCoordinate<GridX>(current_index)) << "\n";
    }

    return 0;
}
