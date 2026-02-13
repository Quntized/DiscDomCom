#include <ddc/ddc.hpp>
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <Kokkos_Core.hpp>

struct X {};
struct Y {};
struct Z {};
template<typename T>
void print_type(const char* name) {
    int status;
    char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
    if (status == 0) {
        std::cout << "  " << name << ": " << demangled << std::endl;
        std::free(demangled);
    } else {
        std::cout << "  " << name << ": " << typeid(T).name() << std::endl;
    }
}
int main() {
  Kokkos::ScopeGuard kokkos_scope;
  ddc::DiscreteDomain<X,Y,Z> dom(ddc::DiscreteElement<X,Y,Z>(0,0,0), ddc::DiscreteVector<X,Y,Z>(4,5,3));
  ddc::Chunk<double,ddc::DiscreteDomain<X,Y,Z>> data("Data", dom);
  ddc::for_each(dom,[&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    data(elem) = i + j + k;
  });
  auto span = data.span_view();
  using SpanType = decltype(span);
  print_type<SpanType>("using SpanType = decltype(span=data.span_view())");
  print_type<decltype(data)>("decltype(data)");
  using spantype = SpanType::span_type;
  print_type<spantype>("SpanType::span_type");
  spantype  span_copy = span;
  std::cout<<"span.data_handle() = "<<span.data_handle()<<std::endl;
  std::cout<<"span_copy.data_handle() = "<<span_copy.data_handle()<<std::endl;
  std::cout << "  Same? " << (span.data_handle() == span_copy.data_handle() ? "YES" : "NO") << std::endl;
  using view_type = SpanType::view_type;
  print_type<view_type>("SpanType::view_type");
  view_type const_view = span.span_cview();
  std::cout<<"const_view(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<const_view(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  using discrete_domain_type = SpanType::discrete_domain_type;
  discrete_domain_type domm = span.domain();
  std::cout<<"discrete_domain_type span_domain = span.domain() has size = "<<domm.size()<<std::endl;
  std::cout<<"It has the front , span_domain.front() = "<<domm.front()<<std::endl;
  std::cout<< " span_domain.back() = "<<domm.back() <<std::endl;
  using memsp = SpanType::memory_space;
  print_type<memsp>(" SpanType::memory_space  ");
  using allocation_mdspan = SpanType::allocation_mdspan_type;
  print_type<allocation_mdspan>("  SpanType::allocation_mdspan_type = ");
  allocation_mdspan mdspan = span.allocation_mdspan();
  std::cout<<" mdspan rank = "<<mdspan.rank()<<std::endl;
  std::cout<<" mdspan size = "<<mdspan.size()<<std::endl;
  std::cout<<" mdspan(1,2,2) = "<<mdspan(1,2,1)<<std::endl;
  mdspan(1,2,1) = 343.0;
  std::cout<<" after mdspan(1,2,1) = 343.0 . so mdspan(1,2,1) = "<<mdspan(1,2,1) <<std::endl;
  std::cout<<" Span(1,2,1)  = "<<span(ddc::DiscreteElement<X,Y,Z>(1,2,1))<<std::endl;
  using const_al_mdspan = SpanType::const_allocation_mdspan_type;
  print_type<const_al_mdspan>(" SpanType::const_allocation_mdspan_type");
  
  const auto& const_mdspan = span;
  const_al_mdspan cnst_mdspn = span.allocation_mdspan();
  std::cout<<"i used constant_allocation_mdspan_type = cnst_mdspn(1,2,1) = "<<cnst_mdspn(1,2,1)<<std::endl;
  using discrete_element_type = SpanType::discrete_element_type;
  print_type<discrete_element_type>("   SpanType::discrete_element_type ");
  discrete_element_type elem(1,2,1);
  std::cout<<" discrete_element_type(1,2,1) = "<<span(elem)<<std::endl;
  using discretevectortype = SpanType::discrete_vector_type;
  discretevectortype dvt(1,2,1);
  std::cout<<" DiscreteVectorType(1,2,1) = "<<span(dvt)<<std::endl;
  discretevectortype extnt = span.domain().extents();
  std::cout<<" Domain extents  = "<<extnt<<std::endl;
  using extents_type = SpanType::extents_type;
  print_type<extents_type>(" SpanType::extents_type  ");
  auto extent  = mdspan.extents();
  std::cout<<"mdspan.rank() =     "<<extent.rank()<<std::endl;
  std::cout<<"mdspan.extent(0) =     "<<extent.extent(0)<<std::endl;
  std::cout<<"mdspan.extent(1) =     "<<extent.extent(1)<<std::endl;
  std::cout<<"mdspan::extent(2) =    "<<extent.extent(2)<<std::endl;
  using layout_type = SpanType::layout_type;
  print_type<layout_type>("layout_type");
  using accessor_type  =  SpanType::accessor_type;
  print_type<accessor_type>("accessor_type");
  accessor_type acc = span.accessor();
  using mapping_type = SpanType::mapping_type;
  mapping_type map = span.mapping();
  print_type<mapping_type>("mapping_type");
  std::cout<<" map.required_span_size() =  "<<map.required_span_size()<<std::endl;
  std::cout << "  is_unique(): " << (map.is_unique() ? "true" : "false") << std::endl;
  std::cout << "  is_exhaustive(): " << (map.is_exhaustive() ? "true" : "false") << std::endl;
  std::cout << "  is_strided(): " << (map.is_strided() ? "true" : "false") << std::endl;
  using element_type = SpanType::element_type;
  print_type<element_type>("element_type");
  using value_type = SpanType::value_type;
  print_type<value_type>("value_type");
  value_type val = 42.5;
  std::cout << "  Created value_type variable: " << val << std::endl;
  using size_type = SpanType::size_type;
  print_type<size_type>("size_type");
  size_type size = span.size();
  std::cout << "  span.size() = " << size << std::endl;
  using data_handle_type = SpanType::data_handle_type;
  print_type<data_handle_type>("data_handle_type");
  data_handle_type ptr = span.data_handle();
  std::cout << "  span.data_handle() = " << ptr << std::endl;
  std::cout << "  ptr[0] = " << ptr[0] << std::endl;
  std::cout << "  ptr[1] = " << ptr[1] << std::endl;
  using ref = SpanType::reference;
  print_type<ref>("reference");
  ref re = span(ddc::DiscreteElement<X,Y,Z>(1,1,1));
  std::cout<<"  SpanType::reference ref = span(ddc::DiscreteElement<X,Y,Z>(1,1,1) ====  "<<re<<std::endl;
  re = 99.9;
  std::cout << "  After ref = 99999:9" << std::endl;
  std::cout << "    span(1,1,1) = " << span(ddc::DiscreteElement<X,Y,Z>(1,1,1)) << std::endl;
  std::cout << "  Reference modifies the span!" << std::endl;
  discrete_domain_type practical_dom = span.domain();
  size_type total_elements = span.size();
  data_handle_type data_ptr = span.data_handle();
  allocation_mdspan md = span.allocation_mdspan();
  std::cout << "Processing " << total_elements << " elements" << std::endl;
  ddc::for_each(practical_dom, [&](discrete_element_type e) {
      ref r = span(e);
      r *= 2.0; 
  });
    
  std::cout << "After doubling all values:" << std::endl;
  std::cout << "  span(0,0,0) = " << span(ddc::DiscreteElement<X,Y,Z>(0,0,0)) << std::endl;
  std::cout << "  span(1,1,1) = " << span(ddc::DiscreteElement<X,Y,Z>(1,1,1)) << std::endl;
  std::cout << "  span(2,2,2) = " << span(ddc::DiscreteElement<X,Y,Z>(2,2,2)) << std::endl;
    
  return 0;
}
