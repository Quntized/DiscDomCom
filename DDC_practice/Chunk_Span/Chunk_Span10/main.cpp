#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>

struct X {};
struct Y {};

int main() {
  Kokkos::ScopeGuard kokkos_scope;
  ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0), ddc::DiscreteVector<X,Y>(3,4));
  Kokkos::View<double**, Kokkos::HostSpace> kview("view", 3,4);
  ddc::ChunkSpan<double, ddc::DiscreteDomain<X,Y>> span1(kview,dom);
  std::cout<<"span1(ddc::DiscreteElement<X,Y>(1,2))   =   "<<span1(ddc::DiscreteElement<X,Y>(1,2))<<std::endl;
  ddc::Chunk<double, ddc::DiscreteDomain<X, Y>> chunk("Chunk", dom);
  chunk(ddc::DiscreteElement<X,Y>(1,2)) = 42.0;
  ddc::ChunkSpan span2(kview, dom);
  std::cout << "2. From Chunk&: " << span2(ddc::DiscreteElement<X,Y>(1,2)) << std::endl;
  const auto& const_chunk = chunk;
  ddc::ChunkSpan span3(const_chunk);
  std::cout<<"span3(ddc::DiscreteElement<X,Y>(1,2))   =  "<<span3(ddc::DiscreteElement<X,Y>(1,2))<<std::endl;
  ddc::ChunkView<double, ddc::DiscreteDomain<X, Y>> view = chunk.span_cview();
  std::cout << "4. ChunkView (const): " << view(ddc::DiscreteElement<X,Y>(1,2)) << std::endl;
  return 0;
}
