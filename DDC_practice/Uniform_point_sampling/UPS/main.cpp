#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

#include <ddc/ddc.hpp>
#include <Kokkos_Core.hpp>
struct X;   
struct Y;   
struct DDimX : ddc::UniformPointSampling<X> {};
struct DDimY : ddc::UniformPointSampling<Y> {};
int main(int argc, char** argv)
{
    ddc::ScopeGuard const ddc_scope(argc, argv);

    {
        ddc::DiscreteDomain<DDimX> const x_domain
                = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
                        ddc::Coordinate<X>(0.0),         
                        ddc::Coordinate<X>(1.0),         
                        ddc::DiscreteVector<DDimX>(5)));  
        std::cout << "  Step size: " << ddc::step<DDimX>() << "\n";
        std::cout << "  Domain size: " << x_domain.size() << "\n";
        std::cout << "  Front index: " << x_domain.front().uid<DDimX>() << "\n";
        std::cout << "  Back index:  " << x_domain.back().uid<DDimX>() << "\n";
        std::cout << "\n  Coordinate mapping:\n";
        ddc::for_each(x_domain, [](ddc::DiscreteElement<DDimX> ix) {
            ddc::Coordinate<X> x = ddc::coordinate(ix);
            std::cout << "    ix=" << ix.uid<DDimX>()
                      << "  ->  x=" << ddc::get<X>(x) << "\n";
        });
    }
    std::cout << "\n";
    {
        auto const [y_main, y_ghosted, y_pre_ghost, y_post_ghost]
                = ddc::init_discrete_space<DDimY>(DDimY::init_ghosted<DDimY>(
                        ddc::Coordinate<Y>(0.0),        
                        ddc::Coordinate<Y>(4.0),         
                        ddc::DiscreteVector<DDimY>(5),   
                        ddc::DiscreteVector<DDimY>(2))); 

        std::cout << "  Step: " << ddc::step<DDimY>() << "\n";

        std::cout << "  Main:    size=" << y_main.size()
                  << ", indices [" << y_main.front().uid<DDimY>()
                  << ".." << y_main.back().uid<DDimY>() << "]\n";
        std::cout << "  Ghosted: size=" << y_ghosted.size()
                  << ", indices [" << y_ghosted.front().uid<DDimY>()
                  << ".." << y_ghosted.back().uid<DDimY>() << "]\n";
        std::cout << "  Pre-ghost:  size=" << y_pre_ghost.size()
                  << ", indices [" << y_pre_ghost.front().uid<DDimY>()
                  << ".." << y_pre_ghost.back().uid<DDimY>() << "]\n";
        std::cout << "  Post-ghost: size=" << y_post_ghost.size()
                  << ", indices [" << y_post_ghost.front().uid<DDimY>()
                  << ".." << y_post_ghost.back().uid<DDimY>() << "]\n";
        std::cout << "\n  All points (ghosted domain):\n";
        ddc::for_each(y_ghosted, [&](ddc::DiscreteElement<DDimY> iy) {
            ddc::Coordinate<Y> y = ddc::coordinate(iy);
            bool in_main = y_main.contains(iy);
            std::cout << "    iy=" << iy.uid<DDimY>()
                      << "  ->  y=" << std::setw(5) << ddc::get<Y>(y)
                      << (in_main ? "  (main)" : "  (ghost)") << "\n";
        });
    }
    std::cout << "\n";
    {


        std::cout << "  DDimX: step = " << ddc::step<DDimX>() << "\n";
        std::cout << "  DDimY: step = " << ddc::step<DDimY>() << "\n";
        ddc::DiscreteDomain<DDimX> dx(
                ddc::DiscreteElement<DDimX>(0), ddc::DiscreteVector<DDimX>(1));
        ddc::DiscreteDomain<DDimY> dy(
                ddc::DiscreteElement<DDimY>(0), ddc::DiscreteVector<DDimY>(1));
        std::cout << "  DDimX: origin = " << ddc::get<X>(ddc::coordinate(dx.front())) << "\n";
        std::cout << "  DDimY: origin = " << ddc::get<Y>(ddc::coordinate(dy.front())) << "\n";
        ddc::DiscreteElement<DDimX> ix3(3);
        double const expected = 0.0 + (3 - 0) * 0.25;
        std::cout << "\n  Formula check for DDimX, ix=3:\n";
        std::cout << "    m_origin + (ix - front) * step\n";
        std::cout << "    = 0.0 + (3 - 0) * 0.25\n";
        std::cout << "    = " << expected << "\n";
        std::cout << "    coordinate(3) = "
                  << ddc::get<X>(ddc::coordinate(ix3)) << "  (matches!)\n";
    }
    std::cout << "\n";

    {
        ddc::DiscreteDomain<DDimX> x_dom(
                ddc::DiscreteElement<DDimX>(0), ddc::DiscreteVector<DDimX>(5));
        ddc::Chunk f_x("f_x", x_dom, ddc::HostAllocator<double>());
        ddc::for_each(x_dom, [&](ddc::DiscreteElement<DDimX> ix) {
            double const x = ddc::get<X>(ddc::coordinate(ix));
            f_x(ix) = std::sin(2.0 * M_PI * x);
        });
        ddc::for_each(x_dom, [&](ddc::DiscreteElement<DDimX> ix) {
            double const x = ddc::get<X>(ddc::coordinate(ix));
            std::cout << "    x=" << std::setw(4) << std::setprecision(2) << x
                      << "  ->  f(x)=" << std::setw(9) << std::setprecision(5)
                      << f_x(ix) << "\n";
        });

   }
   return 0;
}
