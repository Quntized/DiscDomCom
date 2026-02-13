#include <ddc/ddc.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>

void p(auto e,const char* info){
  std::cout<<" =====  "<<e<<"   .  "<<info<<std::endl;
}

struct X{};
struct Y{};
int main() {
  Kokkos::ScopeGuard scope_guard;
  ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0), ddc::DiscreteVector<X,Y>(4,5));
  ddc::Chunk<double , ddc::DiscreteDomain<X,Y>> chunk("MyChunk",dom);
  ddc::for_each(dom , [&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    chunk(elem) = i + j;
  });
  auto mut_chunk_span =  chunk.span_view();
  p(mut_chunk_span(ddc::DiscreteElement<X,Y>(1,1)),"mut_chunk_span = chunk.span_view(), and then at elem(1,1) should 2.");
  ddc::ChunkSpan<const double,ddc::DiscreteDomain<X,Y>> const_span(mut_chunk_span);
  //ddc::ChunkSpan<float,ddc::DiscreteDomain<X,Y>> const_span_f(mut_chunk_span);
  ddc::Chunk<float,ddc::DiscreteDomain<X,Y>> f_chunk("FLoat_chunk",dom);
  auto float_span = f_chunk.span_view();
  ddc::for_each(dom , [&](auto elem){
    float_span(elem) = static_cast<float>(mut_chunk_span(elem));
  });
  auto s = (const_span.data_handle() == mut_chunk_span.data_handle() ? "True ": " False ");
  p(s,"  onst_span.data_handle() == mut_chunk_span.data_handle() " );
  mut_chunk_span(ddc::DiscreteElement<X,Y>(1,1)) = 56;
  p(  mut_chunk_span(ddc::DiscreteElement<X,Y>(1,1)) ,"    mut_chunk_span(ddc::DiscreteElement<X,Y>(1,1)) = 56;  ");
  p(  const_span(ddc::DiscreteElement<X,Y>(1,1)) ,"    const_span(ddc::DiscreteElement<X,Y>(1,1)) = 56;  ");
  //passing spans to functiom;
  auto read_only_functions = [](ddc::ChunkSpan<const double,ddc::DiscreteDomain<X,Y>> cs){
    p(cs(ddc::DiscreteElement<X,Y>(1,1)),"    cs(ddc::DiscreteElement<X,Y>(1,1)) inside function ");
    //s(ddc::DiscreteElement<X,Y>(1,1)) = 1 No can't do it ;
  };
  //read_only_functions(mut_chunk_span); ===== this won't work cause implicit conversion won't work , we need explicit conversion . fro example:
  read_only_functions(ddc::ChunkSpan<const double , ddc::DiscreteDomain<X,Y>>(mut_chunk_span));
  auto compute_sum = [](ddc::ChunkSpan<const double , ddc::DiscreteDomain<X,Y>> span){
    double sum = 0.0;
    ddc::for_each(span.domain(),[&](auto elem){
      sum += span(elem);
    });
    return sum;
  };
  auto ssum = compute_sum(ddc::ChunkSpan<const double ,ddc::DiscreteDomain<X,Y>>(mut_chunk_span));
  p(ssum , "sum of domain  didn't changed the 54 value at 1,1. ");
  
  return 0;
}
