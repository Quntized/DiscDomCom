#include <ddc/ddc.hpp>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>

struct R; 
struct Z;

struct GridR : ddc::UniformPointSampling<R> {};
struct GridZ : ddc::UniformPointSampling<Z> {};

using IdxR = ddc::DiscreteElement<GridR>;
using IdxZ = ddc::DiscreteElement<GridZ>;
using IdxRangeRZ = ddc::DiscreteDomain<GridR, GridZ>;

const double R_major = 2.0; 
const double a_minor = 0.5; 

template <class SpanType>
void init_plasma_profile(SpanType density_view) {
    
    auto domain = density_view.domain();
    //std::cout<<"density_view that you created using r z domain size = "<<domain<<std::endl;

    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        domain,
        KOKKOS_LAMBDA(auto idx) {
            IdxR ir(idx);
            IdxZ iz(idx);
            std::cout<<"ir = "<<ir<<", iz = "<<iz<<std::endl;
            double r_pos = ddc::coordinate(ir);
            double z_pos = ddc::coordinate(iz);
            std::cout<<"r_pos = "<<r_pos<<", z_pos = "<<z_pos<<std::endl;

            double rho_sq = (r_pos - R_major) * (r_pos - R_major) + (z_pos * z_pos);
            
            if (rho_sq < (a_minor * a_minor)) {
                density_view(idx) = 1e19 * (1.0 - (rho_sq / (a_minor * a_minor)));
            } else {
                density_view(idx) = 0.0; 
            }
        }
    );
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    ddc::ScopeGuard ddc_guard(argc, argv);

    std::cout << "HI!!!! Initializing Tokamak Grid " << std::endl;
    auto const dom_r = ddc::init_discrete_space<GridR>(
        GridR::init<GridR>(
            ddc::Coordinate<R>(1.4), 
            ddc::Coordinate<R>(2.6), 
            ddc::DiscreteVector<GridR>(10)
        )
    );
    
    auto const dom_z = ddc::init_discrete_space<GridZ>(
        GridZ::init<GridZ>(
            ddc::Coordinate<Z>(-0.6), 
            ddc::Coordinate<Z>(0.6), 
            ddc::DiscreteVector<GridZ>(10)
        )
    );

    IdxRangeRZ tokamak_domain(dom_r, dom_z);
    ddc::Chunk<double, IdxRangeRZ, ddc::DeviceAllocator<double>> density_ne(tokamak_domain);
    init_plasma_profile(density_ne.span_view());

    Kokkos::fence(); 
    IdxR r_start = dom_r.front() + 7;
    ddc::DiscreteVector<GridR> r_width(3);
    ddc::DiscreteDomain<GridR> r_sub_domain(r_start, r_width);

    auto pedestal_view = density_ne[r_sub_domain];

    std::cout << "\n--- Starting GPU Analysis of Pedestal Region ---\n";
    std::cout << "Scanning " << pedestal_view.size() << " points..." << std::endl;

    ddc::parallel_for_each(
        Kokkos::DefaultExecutionSpace(),
        pedestal_view.domain(),
        KOKKOS_LAMBDA(auto idx) {
            
            double value = pedestal_view(idx);

            if (value > 0.1e19) {
                 IdxR ir(idx);
                 IdxZ iz(idx);
                 double r_val = ddc::coordinate(ir);
                 printf("GPU Found High Density at R=%.3f (Idx %ld): %e\n", 
                        r_val, (long)ir.uid(), value);
            }
        }
    );

    Kokkos::fence();
    std::cout << "\nAnalysis Complete " << std::endl;

    return 0;
}
