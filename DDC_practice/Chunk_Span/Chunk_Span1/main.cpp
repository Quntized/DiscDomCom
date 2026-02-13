#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>

struct X{};
struct Y{};
void p(auto j,const char* info){
  std::cout<<" =====  "<<j<<"   .  "<<info<<std::endl;
}
auto make_temporary_chunk(){
  ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0),ddc::DiscreteVector<X,Y>(3,3));
  ddc::Chunk<double,ddc::DiscreteDomain<X,Y>> temp_chunk("temp_chun",dom);
  return temp_chunk;
}

int main() {
  ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0),ddc::DiscreteVector<X,Y>(3,3));
  ddc::Chunk<double, ddc::DiscreteDomain<X,Y>> chunk("Mychunk", dom);
  ddc::ChunkSpan<double , ddc::DiscreteDomain<X,Y>> chunk_span(chunk);
  p(chunk_span.domain().size(),"chunk_span.domain().size()");
  {
    ddc::Chunk<float , ddc::DiscreteDomain<X,Y>> f_chunk("float_chunk",dom);
    auto f_chunk_span = f_chunk.span_view();
    p(f_chunk_span.domain().size(),"float_chunk.domain().size() ");
  }
  {
    auto t_chunk = make_temporary_chunk();
    auto span_t_chunk = t_chunk.span_view();
    p(span_t_chunk.domain().size(), "temporary_chunk_span.span_view().domain().size()");
  }
  {
    //ddc::ChunkSpan<double,ddc::DiscreteDomain<X,Y>> chun(make_temporary_chunk());
  }
  return 0;
}


