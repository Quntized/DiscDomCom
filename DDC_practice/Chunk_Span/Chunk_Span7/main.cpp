#include <ddc/ddc.hpp>
#include <iostream>
void p(auto e,const char* info){
  std::cout<<" =====  "<<e<<"   .  "<<info<<std::endl;
}

struct X{};
struct Y{};
struct Z{};
template<typename T>
void print_info(const char* name, T value) {
    std::cout << "  " << name << " = " << value << std::endl;
}
int main() {
    Kokkos::ScopeGuard kokkos_scope;
    ddc::DiscreteDomain<X, Y, Z> dom(
        ddc::DiscreteElement<X, Y, Z>(0, 0, 0),
        ddc::DiscreteVector<X, Y, Z>(4, 3, 5)
    );
    ddc::Chunk<double, ddc::DiscreteDomain<X, Y, Z>> chunk("Chunk3D", dom);
    ddc::for_each(dom, [&](auto elem) {
        auto i = ddc::select<X>(elem).uid();
        auto j = ddc::select<Y>(elem).uid();
        auto k = ddc::select<Z>(elem).uid();
        chunk(elem) = i * 100 + j * 10 + k;
    });
    auto span3d = chunk.span_view();
    print_info("Domain size", span3d.domain().size());
    print_info("span3d(2,1,3)", span3d(ddc::DiscreteElement<X,Y,Z>(2,1,3)));
    auto slice_x2 = span3d[ddc::DiscreteElement<X>(2)];
    print_info("slice_x2 domain size", slice_x2.domain().size());
    print_info("slice_x2(1,3)", slice_x2(ddc::DiscreteElement<Y,Z>(1,3)));
    std::cout << "  span3d(2,1,3)  = " << span3d(ddc::DiscreteElement<X,Y,Z>(2,1,3)) << std::endl;
    std::cout << "  span3d(2,1,3)  = " << span3d(ddc::DiscreteElement<X,Y,Z>(2,1,3)) << std::endl;
    std::cout << "  slice_x2(1,3)  = " << slice_x2(ddc::DiscreteElement<Y,Z>(1,3)) << std::endl;
    auto slice_x2_y1 = span3d[ddc::DiscreteElement<X,Y>(2, 1)];
    print_info("slice_x2_y1(3)", slice_x2_y1(ddc::DiscreteElement<Z>(3)));
    std::cout << "  span3d(2,1,3)     = " << span3d(ddc::DiscreteElement<X,Y,Z>(2,1,3)) << std::endl;
    auto slice_y1 = span3d[ddc::DiscreteElement<Y>(1)];
    print_info("slice_y1(2,3)", slice_y1(ddc::DiscreteElement<X,Z>(2,3)));
    auto slice_z3 = span3d[ddc::DiscreteElement<Z>(3)];
    print_info("slice_z3(2,1)", slice_z3(ddc::DiscreteElement<X,Y>(2,1)));
    std::cout << "  span3d(2,1,3) = " << span3d(ddc::DiscreteElement<X,Y,Z>(2,1,3)) << std::endl;
    slice_x2_y1(ddc::DiscreteElement<Z>(3)) = 9999.0;
    std::cout << "  span3d(2,1,3)      = " << span3d(ddc::DiscreteElement<X,Y,Z>(2,1,3)) << std::endl;
    auto double_slice = span3d[ddc::DiscreteElement<X,Y>(1,2)];
    print_info("double_slice(4)", double_slice(ddc::DiscreteElement<Z>(4)));
    auto step1 = span3d[ddc::DiscreteElement<X>(1)];
    auto step2 = step1[ddc::DiscreteElement<Y>(2)];
    print_info("step2(4)", step2(ddc::DiscreteElement<Z>(4)));
    for (std::size_t x = 0; x < 4; ++x) {
        auto x_plane = span3d[ddc::DiscreteElement<X>(x)];
        double sum = 0.0;
        ddc::for_each(x_plane.domain(), [&](auto elem) {
            sum += x_plane(elem);
        });
        std::cout << "  Sum of X=" << x << " plane: " << sum << std::endl;
    }
    
    return 0;
    }
    
