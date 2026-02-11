#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>

struct X {};
struct Y {};
struct Z {};

int main() {
  Kokkos::ScopeGuard kokkos_scope;
  ddc::DiscreteDomain<X,Y,Z> dom(ddc::DiscreteElement<X,Y,Z>(0,0,0),ddc::DiscreteVector<X,Y,Z>(3,4,5));
  ddc::Chunk<double, ddc::DiscreteDomain<X,Y,Z>> chunk1("Chunk1", dom);
  std::cout<<" Chunk1.size() = "<<chunk1.size()<<std::endl;
  std::cout<<"Chunk1.data_handle() = "<<chunk1.data_handle()<<std::endl;
  ddc::for_each(dom , [&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    chunk1(elem) = i+j+k;
  });
  std::cout<<"chunk1(0,0,0) = "<<chunk1(ddc::DiscreteElement<X,Y,Z>(0,0,0))<<std::endl;
  std::cout<<"chunk1(3,4,5) = "<<chunk1(ddc::DiscreteElement<X,Y,Z>(2,3,4))<<std::endl;
  std::cout << "\n  Copy construction calls and creates view to same data:" << std::endl;
  auto span1 = chunk1.span_view();
  span1(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = 121.0;
  std::cout<<"First used span1 = Chunk1.span_view , then used span1(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = 121.0 , now finding out Chunk1(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<chunk1(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<"  span1(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<span1(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  std::cout<< "  span1.data_handle()  =   "<<span1.data_handle()<<std::endl;
  std::cout<<"   same pointer? "<<(span1.data_handle() == chunk1.data_handle() ? "Same": "True ")<<std::endl;
  auto span2 = span1;
  span2(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = 1234.0;
  std::cout<<" span2(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<span2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<" span1(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<span1(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<" . chunk1(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<chunk1(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  std::cout<< "chunk1.data_handle() == span2.data_handle() "<< (chunk1.data_handle() == span2.data_handle() ? "Same" : "Not Same")<<std::endl;
  auto const_span = chunk1.span_cview();
  std::cout<<"   const_span(1,1,1) = "<<const_span(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  auto old_ptr = chunk1.data_handle();
  ddc::Chunk<double , ddc::DiscreteDomain<X,Y,Z>> chunk2 = std::move(chunk1);
  std::cout<<" chunk2.data_handle() -------- "<<chunk2.data_handle()<<std::endl;
  std::cout<<"chunk2.data_handle() == chunk1.data_handle() "<<(chunk2.data_handle() == chunk2.data_handle() ? "Same": "No Same")<<std::endl;
  std::cout<<" chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  double* ptr = chunk2.data_handle();
  for(std::size_t i = 0; i<dom.size(); ++i){
    std::cout<<" "<<i <<" "<<ptr[i]<<std::endl;
  }
  auto mdspan = chunk2.allocation_mdspan();
  std::cout<<"mdspan = chunk2.allocation_mdspan() , mdspan(1,1,1) = "<<mdspan(1,1,1)<<std::endl;
  mdspan(1,1,1) = 765.0;
  std::cout<<" Chunk2(1,1,1) = "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<" . mdspan(1,1,1) = "<<mdspan(1,1,1)<<std::endl;
  auto view1 = chunk2.span_view();
  auto span3 =view1;
  std::cout<<"view1 = chunk2.span_view() , span3 = view1. Now span3 == view1 ----- "<<(span3.data_handle() == view1.data_handle() ? "Same Span3 = view1 ": "No same")<<std::endl;
  span3(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = 45454;
  std::cout<<"I did span3(ddc::DiscreteElement<X,Y,Z>(1,1,1) = 45454; Now , chunk2(1,1,1) = "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<< " . span3(1,1,1) = "<<span3(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  auto process_data = [](auto span){
    ddc::for_each(span.domain(),[&](auto elem){
      span(elem) *= 2.0;
    });
  };
  std::cout<<" Before multiplying 2 . The data at (1,1,1) using chunk2(1,1,1) = "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  process_data(chunk2.span_view());
  std::cout<< " After multiplying 2 the data at 1,1,1 . chunk2(1,1,1). "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  auto slice = chunk2[ddc::DiscreteElement<X>(2)];
  std::cout<<" slice domain size , the slice came from slice = chunk2[ddc::DiscreteElement<X>(2)] . slice size"<< slice.domain().size()<<std::endl;
  slice(ddc::DiscreteElement<Y,Z>(1,1)) = 76432.0;
  std::cout<<"I had changed the slice(ddc::DiscreteElement<X,Y,Z>(1,1)) = 76432.0. Now chunk2(1,1,1) = "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<". And slice(1,1) has = "<<slice(ddc::DiscreteElement<Y,Z>(1,1))<<std::endl;
  ddc::Chunk<double, ddc::DiscreteDomain<X,Y,Z>> chunk3("Chunk3",dom);
  ddc::parallel_deepcopy(chunk3,chunk2);
  std::cout<<"Chunk3 == chunk2 ? let's see: "<<(chunk3.data_handle() == chunk2.data_handle() ? "Same": "Not Same")<<std::endl;
  std::cout<<"Before modifying chunk3 : chunk2 = "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<". Chunk3(1,1,1) = "<<chunk3(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  chunk3(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = 11.0;
  std::cout<<"after modifying chunk3 , chunk2(1,1,1)  =  "<<chunk2(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<" . Chunk3(1,1,1) = "<<chunk3(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  
  
  return 0;
}
