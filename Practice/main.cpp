#include <iostream>
#include <tuple>
#include <ddc/ddc.hpp>
#include <ddc/uniform_point_sampling.hpp>

struct DimX;
using DDimX = ddc::UniformPointSampling<DimX>;

int main() {
    double x_start = 0.0;
    double x_end = 10.0;
    int num_points = 50;

    auto const result = DDimX::init<DDimX>(
        ddc::Coordinate<DimX>(x_start), 
        ddc::Coordinate<DimX>(x_end), 
        ddc::DiscreteVector<DDimX>(num_points)
    );

    auto& [grid_impl, domain_x] = result;

    std::cout << "Discretization Initialized!" << std::endl;
    std::cout << "Number of points: " << domain_x.size() << std::endl;

    for (ddc::DiscreteElement<DDimX> index : domain_x) {
        // FIXED LINE BELOW: Use the member function of the implementation object
        ddc::Coordinate<DimX> coord = grid_impl.coordinate(index);

        std::cout << "Index: " << index.uid() 
                  << " -> Coordinate: " << coord 
                  << std::endl;
    }

    return 0;
}
