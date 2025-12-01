#include <ddc/ddc.hpp>
#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp> // Still needed for DDC internals

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
    ddc::for_each(domain, [&](auto idx) {
        IdxR ir(idx);
        IdxZ iz(idx);
        std::cout<<"element r = "<<ir<<" . Element z = "<<iz<<std::endl;
        double r_pos = ddc::coordinate(ir);
        double z_pos = ddc::coordinate(iz);
        std::cout << "r pos: " << r_pos << ", z pos: " << z_pos << std::endl;
        double rho_sq = (r_pos - R_major) * (r_pos - R_major) + (z_pos * z_pos);
        if (rho_sq < (a_minor * a_minor)) {
            density_view(idx) = 1e19 * (1.0 - (rho_sq / (a_minor * a_minor)));
        } else {
            density_view(idx) = 0.0; 
        }
    });
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    ddc::ScopeGuard ddc_guard(argc, argv);

    std::cout << "Initializing Tokamak Grid (CPU)" << std::endl;
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
    ddc::Chunk<double, IdxRangeRZ> density_ne(tokamak_domain);
    init_plasma_profile(density_ne.span_view());
    int n_points_to_scan = 3; 
    IdxR r_start = dom_r.back() - (n_points_to_scan - 1);
    ddc::DiscreteVector<GridR> r_width(n_points_to_scan);
    ddc::DiscreteDomain<GridR> r_sub_domain(r_start, r_width);
    auto pedestal_view = density_ne[r_sub_domain];
    std::cout << "\n In CPU Pedestal Region ---\n";
    std::cout << "Scanning " << pedestal_view.size() << " points..." << std::endl;
    ddc::for_each(pedestal_view.domain(), [&](auto idx) {
        
        double value = pedestal_view(idx);

        if (value <0.1) {
             IdxR ir(idx);
             IdxZ iz(idx);
            std::cout<<"element r = "<<ir<<" . Element z = "<<iz<<std::endl;
            double r_val = ddc::coordinate(ir);

             
             std::cout << "CPU Found High Density at R=" << r_val 
                       << " (Idx " << ir.uid() << "): " << value << "\n";
        }
    });

    std::cout << "\n Analysis Complete" << std::endl;

    return 0;
}
