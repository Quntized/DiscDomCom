#include <ddc/ddc.hpp>
#include <iostream>
#include <typeinfo>
#include <Kokkos_Core.hpp>

struct X{};
struct Y{};
struct Z{};

/* template<typename T>
void print_type_name(const char* var_name){
  std::cout<<" "<<var_name<<" : "<<typeid(T).name()<<std::endl;
} */
template<typename T>
void print_type_name(const char* var_name) {
    int status;
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
        std::free
    );
    
    if (status == 0) {
        std::cout << "  " << var_name << ": " << demangled.get() << std::endl;
    } else {
        std::cout << "  " << var_name << ": " << typeid(T).name() << std::endl;
    }
}

int main() {
  ddc::DiscreteDomain<X,Y,Z> dom(
    ddc::DiscreteElement<X,Y,Z>(0,0,0),
    ddc::DiscreteVector<X,Y,Z>(2,3,4)
  );
  ddc::Chunk<double , ddc::DiscreteDomain<X,Y,Z>> data("Chunk", dom);
  using ChunkType  = decltype(data);
  using DiscreteElementType = ChunkType::discrete_element_type;
  using DiscreteVectorType = ChunkType::discrete_vector_type;
  using ElementType = ChunkType::element_type;
  using ValueType = ChunkType::value_type;
  using SizeType = ChunkType::size_type;
  print_type_name<ChunkType>("chunk_type");
  print_type_name<DiscreteElementType>("discrete_element_type");
  print_type_name<DiscreteVectorType>("discrete_vector_type");
  print_type_name<ElementType>("element_type");
  print_type_name<ValueType>("value_type");
  std::cout<<"Rank of Chunk type "<<ChunkType::rank()<<std::endl;
  std::cout<<"Dynamic Rank of Chunk type "<<ChunkType::rank_dynamic()<<std::endl;
  for(std::size_t i = 0; i<ChunkType::rank(); ++i){
    auto extent = ChunkType::static_extent(i);
    if (extent == Kokkos::dynamic_extent){
      std::cout<<" Dimension "<<i<<" : dynamic "<<std::endl;
    }else{
      std::cout<<" Dimension "<<i <<" : "<<extent<<std::endl;
    }
  }
  std::cout<<"is_always_unique() ---> "<<(ChunkType::is_always_unique() ? "True": "False")<<std::endl;
  std::cout<<"is_always_exhaustive() ---> "<<(ChunkType::is_always_exhaustive() ? "True": "False")<<std::endl;
  std::cout<<"is_always_strided() ---> "<<(ChunkType::is_always_strided() ? "True": "False")<<std::endl;
  auto mdspan = data.allocation_mdspan();
  std::cout<<"Mdspan properties: "<<std::endl;
  std::cout<<"Rank = "<<mdspan.rank()<<std::endl;
  std::cout<<"Size = "<<mdspan.size()<<std::endl;
  std::cout<<"Extent(0) = "<<mdspan.extent(0)<<std::endl;
  std::cout<<"Extent(1) = "<<mdspan.extent(1)<<std::endl;
  std::cout<<"Extent(2) = "<<mdspan.extent(2)<<std::endl;
  std::cout<<"Strided(0) = "<<mdspan.stride(0)<<std::endl;
  std::cout<<"Strided(1) = "<<mdspan.stride(1)<<std::endl;
  std::cout<<"Strided(2) = "<<mdspan.stride(2)<<std::endl;
  ChunkType::discrete_domain_type chunk_domain = data.domain();
  std::cout<<"ChunkType::discrete_domain_type  = "<<chunk_domain.size()<<std::endl;
  ChunkType::discrete_element_type elem(1,2,1);
  std::cout<<"ChunkType::discrete_element_type  = "<<data(elem)<<std::endl;
  ChunkType::discrete_vector_type vec(1,2,1);
  std::cout<<"ChunkType::discrete_vector_type = "<<data(vec)<<std::endl;
  ChunkType::data_handle_type ptr = data.data_handle();
  std::cout<<"Data_handle pointer = "<<ptr<<std::endl;
  std::cout<<"pointer[0] = "<<ptr[0]<<std::endl;
  ChunkType::reference ref = data(elem);
  std::cout<<"Original value = "<<ref<<std::endl;
  ref= 99.6;
  std::cout<<"Modified via ChunkType::reference  = "<<ref<<std::endl;
  ddc::Chunk<float, ddc::DiscreteDomain<X,Y,Z>> float_data("Float_chunk",dom);
  ddc::Chunk<int, ddc::DiscreteDomain<X,Y,Z>> int_data("int_data",dom);
  std::cout<<"Double Chunk rank: "<<data.rank()<<std::endl;
  std::cout<<"Double Chunk rank using decltype: "<<decltype(data)::rank()<<std::endl;
  print_type_name<decltype(data)>("Using Decltype(data)");
  std::cout<<"Float Chunk rank using decltype: "<<decltype(float_data)::rank()<<std::endl;
  std::cout<<"Int Chunk rank using decltype: "<<decltype(int_data)::rank()<<std::endl;
  using ExtentType = ChunkType::extents_type;
  print_type_name<ExtentType>("using ChunkType::extents_type");
  std::cout<<"============="<<std::endl;
  print_type_name<decltype(mdspan)>("Using Decltype(mdspan) where mdspan = data.allocation_mdspan()");
  auto extents = mdspan.extents();
  print_type_name<decltype(extents)>("Using Decltype(extents)");
  print_type_name<decltype(ExtentType::rank())>("Using Decltype(ExtentType::rank())");
  std::cout<<"Extents object: "<<std::endl;
  for(std::size_t i =0; i<ExtentType::rank(); ++i){
    std::cout<< "extent ( "<<i<<" ) "<<extents.extent(i)<<std::endl;
  }
  using AllocMDSpanType = ChunkType::allocation_mdspan_type;
  print_type_name<AllocMDSpanType>("Using AllocMDSpanType ");
  AllocMDSpanType mdspan_mutable = data.allocation_mdspan();
  mdspan_mutable(1,1,1) = 23.087;
  using ConstAllocMDSpanType = ChunkType::const_allocation_mdspan_type;
  const auto& data_const = data;
  ConstAllocMDSpanType mdspan_const = data_const.allocation_mdspan();
  std::cout<<mdspan_const(1,1,1)<<std::endl;
  using LayoutType = ChunkType::layout_type;
  auto mapping = mdspan.mapping();
  print_type_name<decltype(mdspan.mapping())>("Using decltype(mdspan.mapping()");
  std::cout<<"Layout type information: "<<std::endl;
  std::cout<<" required_span_size() "<<mapping.required_span_size()<<std::endl;
  print_type_name<decltype(mapping.required_span_size())>("Using decltype(mapping.required_span_size())");
  std::cout<<"is_unique() "<<(mapping.is_unique() ? "True" : "False") <<std::endl;
  std::cout<<"is_exhaustive() "<<(mapping.is_exhaustive() ? "True" : "False")<<std::endl;
  std::cout<<"is_strided() "<<(mapping.is_strided() ? "true" : "False")<<std::endl;
  print_type_name<ChunkType::value_type>("Using ChunkType::value_type");
  ddc::for_each(data.domain(), [&](ChunkType::discrete_element_type  elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    data(elem) = static_cast<ChunkType::value_type>(i+j+k);
  });
  ChunkType::discrete_element_type test_elem(1,2,1);
  ChunkType::value_type val = data(test_elem);
  std::cout<<"Value at data(1,2,1) "<<val<<std::endl;
  print_type_name<ChunkType::size_type>(" ChunkType::size_type ");
  ChunkType::size_type total_elements = data.domain().size();
  double sum = 0.0;
  for(ChunkType::size_type i = 0; i<total_elements; ++i){
    sum += data.data_handle()[i];
  }
  ChunkType::value_type mean = sum/total_elements;
  std::cout<<"Mean value : "<< mean<<std::endl;
  
  
return 0;  
}
