#include <iostream>
#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>
struct RadialDim {
    static bool constexpr PERIODIC = false;
};
template <std::size_t Degree>
using RadialBSplines = ddc::NonUniformBSplines<RadialDim, Degree>;
using MyKnots = ddc::NonUniformBsplinesKnots<RadialBSplines<3>>;
int main() {
    std::cout << "SUCCESS: Compilation complete!" << std::endl;
    std::cout << "Basis (Capital S) and Knots (Lowercase s) are mixed." << std::endl;
    std::cout << "Spline Degree: " << RadialBSplines<3>::degree() << std::endl;
    return 0;
}
