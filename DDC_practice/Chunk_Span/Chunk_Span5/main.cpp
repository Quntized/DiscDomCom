#include <ddc/ddc.hpp>
#include <iostream>
#include<Kokkos_Core.hpp>
struct X{};
struct Y{};

void p(auto e,const char* info){
  std::cout<<" =====  "<<e<<"   .  "<<info<<std::endl;
}

int main() {
  Kokkos::ScopeGuard kokkos_scope;
  ddc::DiscreteDomain<X,Y> dom(ddc::DiscreteElement<X,Y>(0,0), ddc::DiscreteVector<X,Y>(5,6));
  ddc::Chunk<double ,ddc::DiscreteDomain<X,Y>> chunk("THIS IS CHUNK",dom);
  ddc::for_each(dom, [&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    chunk(elem) = i+ j ;
  });
  double* ptr = chunk.data_handle();
  ddc::ChunkSpan<double , ddc::DiscreteDomain<X,Y>> span_ptr(ptr,dom);
  p(span_ptr.data_handle(),"     -------> this memory address, of    ddc::ChunkSpan<double , ddc::DiscreteDomain<X,Y>> span_ptr(ptr,dom);    ");
  p( span_ptr(ddc::DiscreteElement<X,Y>(1,1)), "  --------->  this is span_ptr(ddc::DiscreteElement<X,Y>(1,1)) .   ");
  auto d  = (span_ptr(ddc::DiscreteElement<X,Y>(1,1)) == chunk(ddc::DiscreteElement<X,Y>(1,1)) ? "  TRUE   " : "   FALSE  ");
  p(d,"    -------> span_ptr(ddc::DiscreteElement<X,Y>(1,1)) == chunk(ddc::DiscreteElement<X,Y>(1,1)) ?  TRUE     FALSE  ) " );
  constexpr std::size_t array_size = 30;
  double raw_array[array_size];
  for(std::size_t i =0; i<array_size;  ++i){
    raw_array[i] = static_cast<double>(i);
  }
  ddc::ChunkSpan<double, ddc::DiscreteDomain<X,Y>> span_from_array(raw_array,dom);
  auto v = span_from_array(ddc::DiscreteElement<X,Y>(1,1));
  p(v,"  ------------ >   ddc::ChunkSpan<double, ddc::DiscreteDomain<X,Y>> span_from_array(raw_array,dom);           ");
  ddc::for_each(span_from_array.domain(), [&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    p(span_from_array(ddc::DiscreteElement<X,Y>(i,j ))  ,"  ");
  });
  span_from_array(ddc::DiscreteElement<X,Y>(0,0)) = 9999.0;
  p(raw_array[0], " ------------>      raw_array[0]     ");
  std::vector<double> vec(30,42.0);
  p(vec.data(),"  -------> vec.data()");
  ddc::ChunkSpan<double,ddc::DiscreteDomain<X,Y>> vec_span(vec.data(),dom);
  p(vec_span(ddc::DiscreteElement<X,Y>(1,1)), "   ------------>     vec_span(ddc::DiscreteElement<X,Y>(1,1)), " );
  ddc::for_each(dom, [&](auto elem) {
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    vec_span(elem) = i + j * 10;
  });
  p(vec[0],"   ---------> vec[0]");
  p(vec[1], " --------> vec[1]");
  double manual_array[30];
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 6; ++j) {
            int idx = i * 4 + j;  // Row-major: i * num_cols + j
            manual_array[idx] = i * 100 + j;
        }
    }
    ddc::ChunkSpan<double,ddc::DiscreteDomain<X,Y>> span_domain(manual_array, dom);
    p(span_domain(ddc::DiscreteElement<X,Y>(1,1)),"   ------------>  span_domain(ddc::DiscreteElement<X,Y(1,1)>),");
    //you can also do the same with given const at elementtype;
    //auto safe_span = chunk.span_view(); This is safe and recommended ;
    
  return 0;
}
