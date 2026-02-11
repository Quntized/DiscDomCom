#include <iostream>
#include <ddc/ddc.hpp>

struct TagX {};
struct TagY {};

int main() {
    Kokkos::ScopeGuard kokkos_scope;
    
    ddc::DiscreteDomain<TagX, TagY> domain(
        ddc::DiscreteElement<TagX, TagY>(0, 0),
        ddc::DiscreteVector<TagX, TagY>(10, 20)
    );
    
    ddc::Chunk<double, ddc::DiscreteDomain<TagX, TagY>> data("my_data", domain);
    
    // Use ddc::for_each for multi-dimensional iteration
    ddc::for_each(domain, [&](ddc::DiscreteElement<TagX, TagY> elem) {
        auto x = ddc::uid<TagX>(elem);
        auto y = ddc::uid<TagY>(elem);
        data(elem) = x * 10.0 + y;
    });
    
    std::cout << "Domain size: " << domain.size() << " elements\n";
    std::cout << "Value at (5, 10): " << data(ddc::DiscreteElement<TagX, TagY>(5, 10)) << "\n";
    std::cout << "Value at (3, 7): " << data(ddc::DiscreteElement<TagX, TagY>(3, 7)) << "\n";
    
    return 0;
}
