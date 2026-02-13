#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>

struct X {};
struct Y {};

void p(auto e,const char* info){
  std::cout<<" =====  "<<e<<"   .  "<<info<<std::endl;
}

int main() {
  Kokkos::ScopeGuard kokkos_scope;
  ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0),ddc::DiscreteVector<X,Y>(3,4));
  ddc::Chunk<double,ddc::DiscreteDomain<X,Y>> Ddom("Data_domain",dom);
  ddc::for_each(dom, [&](auto elem) {
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    Ddom(elem) = i * 10 + j;
  });
  auto mdspan = Ddom.allocation_mdspan();
  p(mdspan.rank(), " ----------> mdspan.rank()");
  p(mdspan.extent(0), "  -------------> mdspan.extent(0) ");
  p(mdspan.extent(1),"    ----------------> mdspan.extent(1)" );
  p(mdspan(1,2), "  --------------------> mdspan(1,2)");
  ddc::ChunkSpan<double , ddc::DiscreteDomain<X,Y>> span_md(mdspan,dom);
  // Use when: You already have an mdspan;
  p(span_md(ddc::DiscreteElement<X,Y>(1,1)),"   ------>  ddc::DiscreteElement<X,Y>(1,1))");
  auto d = ddc::select<X>(dom.extents()).value();
  p(d,"    --------->  ddc::select<X>(dom.extents()).value())");
  span_md(ddc::DiscreteElement<X,Y>(1,1)) = 23.0;
  p(span_md(ddc::DiscreteElement<X,Y>(1,1)) , "   --------------------> span_md(ddc::DiscreteElement<X,Y>(1,1)). after modifying span_md(ddc::DiscreteElement<X,Y>(1,1)) = 23.0 ");
  p(mdspan(1,1), " -------------> mdspan(0,0),    ddc::ChunkSpan<double , ddc::DiscreteDomain<X,Y>> span_md(mdspan,dom); and   auto mdspan = Ddom.allocation_mdspan();  ");
  Kokkos::View<double** , Kokkos::HostSpace> kokkos_view("kokkos_view", 3,4);
  p(kokkos_view.extent(0),   "    ----------------------->kokkos_view.extent(0) .   Kokkos_View<double** , Kokkos::HostSpace> kokkos_view(kokkos_view, 3,4);");
  p(kokkos_view.extent(1),   "    ----------------------->kokkos_view.extent(1) .   Kokkos_View<double** , Kokkos::HostSpace> kokkos_view(kokkos_view, 3,4);");
  for (std::size_t o = 0; o<kokkos_view.extent(0); ++o){
    for (std::size_t m = 0; m<kokkos_view.extent(1); ++m){
      kokkos_view(o,m) = o + m;
    }
  }
  ddc::ChunkSpan<double, ddc::DiscreteDomain<X,Y>> span_kokkos(kokkos_view,dom);
  
  p(span_kokkos.domain().size(),"  ----------->   span_kokkos.domain().size() . ");
  p(span_kokkos(ddc::DiscreteElement<X,Y>(1,1)) , "  --------> span_kokkos(ddc::DiscreteElement<X,Y>(1,1))");
  //Converts Kokkos::View → mdspan using build_mdspan()
  //ChunkSpan viewing Kokkos::View's memory
  span_kokkos(ddc::DiscreteElement<X,Y>(1,1)) = 12.2;
  p(kokkos_view(1,1), " ----------> kokkos_view(1,1),");
  auto kokkos_computation = [](){
    Kokkos::View<double**, Kokkos::HostSpace> result("Result", 3,4);
    for(std::size_t i = 0; i<result.extent(0); ++i){
      for(std::size_t j = 0; j < result.extent(1); ++j){
        result(i,j) = i + j;
      }
    }
    return result;
  };
  auto kokkos_result = kokkos_computation();
  ddc::ChunkSpan<double,ddc::DiscreteDomain<X,Y>> kokkos_span(kokkos_result, dom);
  double sum = 0.0;
  ddc::for_each(kokkos_span.domain(), [&](auto elem){
    sum += kokkos_span(elem);
  });
  p(sum , "  ------____>     sum += kokkos_span(elem); ");

  return 0;
}
