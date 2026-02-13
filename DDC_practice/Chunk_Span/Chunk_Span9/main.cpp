#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>
struct X {};
struct Y {};

int main() {
  Kokkos::ScopeGuard kokkos_scope;
  ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0), ddc::DiscreteVector<X,Y>(3,5));
  ddc::Chunk<double, ddc::DiscreteDomain<X,Y>> DD("Doom ", dom);
  ddc::for_each(DD.domain(), [&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    DD(elem) = i + j;
  });
  auto span = DD.span_view();
  double* ptr = span.data_handle();
  std::cout << "ptr[0] = " << ptr[0] << std::endl;
  std::cout << "ptr[5] = " << ptr[5] << std::endl;
  auto mdspan = span.allocation_mdspan();
  std::cout << "mdspan.rank() = " << mdspan.rank() << std::endl;
  std::cout << "mdspan(1,2) = " << mdspan(1, 2) << std::endl;
  auto view = span.allocation_kokkos_view();
  std::cout << "view.extent(0) = " << view.extent(0) << std::endl;
  std::cout << "view.extent(1) = " << view.extent(1) << std::endl;
  std::cout << "view(1,2) = " << view(1, 2) << std::endl;
  std::cout << "span(1,2)   = " << span(ddc::DiscreteElement<X,Y>(1,2)) << std::endl;
  std::cout << "mdspan(1,2) = " << mdspan(1, 2) << std::endl;
  std::cout << "view(1,2)   = " << view(1, 2) << std::endl;
  return 0;
}
