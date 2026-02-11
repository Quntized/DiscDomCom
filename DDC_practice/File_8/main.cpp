#include <ddc/ddc.hpp>
#include <iostream>
struct X{};
struct Y{};
struct Z{};

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
    ddc::DiscreteVector<X,Y,Z>(4,3,5)
  );
  ddc::Chunk<double,ddc::DiscreteDomain<X,Y,Z>> data("Data",dom);
  
  ddc::for_each(dom, [&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    data(elem) = i + j + k;
  });
  using ChunkType = decltype(data);
  print_type_name<ChunkType::accessor_type>("ChunkType::accessor");
  auto acc = data.accessor();
  print_type_name<decltype(acc)>("Decltype(accessor)");
  auto extents = data.extents();
  std::cout<<"X extent = "<<ddc::select<X>(extents).value()<<std::endl;
  std::cout<<"Y extent = "<<ddc::select<Y>(extents).value()<<std::endl;
  std::cout<<"Z extent = "<<ddc::select<Z>(extents).value()<<std::endl;
  auto x_extent = data.extent<X>();
  auto y_extent = data.extent<Y>();
  auto z_extent= data.extent<Z>();
  std::cout<<"extent<X> = "<<x_extent<<std::endl;
  std::cout<<"extent<Y> = "<<y_extent<<std::endl;
  std::cout<<"extent<Z> = "<<z_extent<<std::endl;
  auto total_size = data.size();
  std::cout<<"data.size() = "<<total_size<<std::endl;
  auto mapping = data.mapping();
  std::cout<<"required_span_size() = "<<mapping.required_span_size()<<std::endl;
  std::cout << "  This is the minimum memory needed to store the chunk" << std::endl;
  std::cout<<"is_unique() = "<<(data.is_unique() ? "True" : "False") <<std::endl;
  std::cout << "    Each element has unique memory location" << std::endl;
  std::cout<<"is_exhaustive() -- "<<(data.is_exhaustive() ? "True" : "False")<<std::endl;
  std::cout << "     All allocated memory is used (no gaps)" << std::endl;
  std::cout<<"   is_strided() ==== "<<(data.is_strided() ? "True" : "False")<<std::endl;
  std::cout << "     Elements are laid out with regular strides" << std::endl;
  auto x_stride = data.stride<X>();
  auto y_stride = data.stride<Y>();
  auto z_stride = data.stride<Z>();
  std::cout<<"x_stride : "<<x_stride<<std::endl;
  std::cout<<" y_stride : "<<y_stride<<std::endl;
  std::cout<<" z_stride : "<<z_stride<<std::endl;
  std::cout << "\nMemory layout explanation:" << std::endl;
  std::cout << "  To go to next X element: skip " << x_stride << " elements" << std::endl;
  std::cout << "  To go to next Y element: skip " << y_stride << " elements" << std::endl;
  std::cout << "  To go to next Z element: skip " << z_stride << " element" << std::endl;
  auto full_domain = data.domain();
  std::cout<<" domain.size() = "<<full_domain.size()<<std::endl;
  print_type_name<decltype(full_domain.size())>("decltype(full_domain.size())");
  print_type_name<decltype(data.size())>("decltype(data.size())");
  print_type_name<decltype(full_domain)>("decltype(full_domain)");
  print_type_name<decltype(data)>("decltype(data)");
  std::cout<<"full_domain.front() = "<<full_domain.front()<<std::endl;
  std::cout<<"ddc::select full_domain<X> = "<<ddc::select<X>(full_domain.front())<<std::endl;
  std::cout<<"ddc::select full_domain<Y> = "<<ddc::select<Y>(full_domain.front())<<std::endl;
  std::cout<<"ddc::select full_domain<Z> = "<<ddc::select<Z>(full_domain.front())<<std::endl;
  std::cout<<"ddc::select full_domain<X> = "<<ddc::select<X>(full_domain.front()).uid()<<std::endl;
  std::cout<<"ddc::select full_domain<Y> = "<<ddc::select<Y>(full_domain.front()).uid()<<std::endl;
  std::cout<<"ddc::select full_domain<Z> = "<<ddc::select<Z>(full_domain.front()).uid()<<std::endl;
  print_type_name<decltype(full_domain.front())>("decltype(full_domain.front())");
  print_type_name<decltype(ddc::select<X,Y,Z>(full_domain.front()))>("decltype(ddc::select<X,Y,Z>(full_domain.front()))");
  print_type_name<decltype(ddc::select<X,Y>(full_domain.front()))>("decltype(ddc::select<X,Y>(full_domain.front()))");
  auto i = ddc::select<X>(full_domain.front()).uid();
  print_type_name<decltype(i)>("decltype(ddc::select<X>(full_domain.front()).uid())");
  print_type_name<decltype(ddc::select<X>(extents))>("decltype(ddc::select<X>(ddc::select<X>(extent))");
  print_type_name<decltype(data.size())>("decltype(data.size())");
  auto x_domain = data.domain<X>();
  auto y_domain = data.domain<Y>();
  auto z_domain = data.domain<Z>();
  std::cout<<"x_domain =  "<< x_domain.size()<<std::endl;
  std::cout<<"y_domain =  "<<y_domain.size()<<std::endl;  
  std::cout<<"z_domain =  "<<z_domain.size()<<std::endl;
  auto xy_domain = data.domain<X,Y>();
  auto yz_domain = data.domain<Y,Z>();
  auto xz_domain = data.domain<X,Z>();
  std::cout<<"xy_domain =  "<<xy_domain.size()<<std::endl;
  std::cout<<"yz_domain =  "<<yz_domain.size()<<std::endl;  
  std::cout<<"xz_domain =  "<<xz_domain.size()<<std::endl;
  ddc::for_each(data.domain<Y>(), [&] (auto elem){
    auto j = ddc::select<Y>(elem).uid();
    std::cout<<"     Y       =       "<<j<<std::endl;
  });
  auto dims = data.extents();
  auto nx = ddc::select<X>(dims).value();
  auto ny = ddc::select<Y>(dims).value();
  auto nz = ddc::select<Z>(dims).value();
  std::cout << "Grid dimensions: " << nx << " × " << ny << " × " << nz << std::endl;
  std::cout << "Middle point: (" << nx/2 << ", " << ny/2 << ", " << nz/2 << ")" << std::endl;
  auto middle = ddc::DiscreteElement<X,Y,Z>(nx/2, ny/2, nz/2);
  std::cout << "Value at middle: " << data(middle) << std::endl;
  auto elem = ddc::DiscreteElement<X,Y,Z>(2, 1, 1);
  auto ii = ddc::select<X>(elem).uid();
  auto jj = ddc::select<Y>(elem).uid();
  auto kk = ddc::select<Z>(elem).uid();
  auto s = ii+jj+kk;
  auto offset= ii*x_stride + jj*y_stride+kk*z_stride;
  std::cout << "Element (" << ii << "," << jj << "," << kk << "):" << std::endl;
  auto* ptr = data.data_handle();
  std::cout << "  Computed offset: " << offset << std::endl;
  std::cout << "  Value via chunk(elem): " << data(elem) << std::endl;
  std::cout<<"Value via pointer offset : "<<ptr[offset]<<std::endl;
  std::cout << "  data.size() = " << data.size() << std::endl;
  std::cout << "  data.domain().size() = " << data.domain().size() << std::endl;
  std::cout << "  extent<X>() × extent<Y>() × extent<Z>() = " 
              << (data.extent<X>() * data.extent<Y>() * data.extent<Z>()) << std::endl;
  std::cout << "  All are equal: " 
            << (data.size() == data.domain().size() ? "OK" : "NO") << std::endl;
  auto slice = data[ddc::DiscreteElement<X>(1)];
std::cout << "After slicing at X=2:" << std::endl;
std::cout << "  Original domain: 3D (" << data.domain().size() << " elements)" << std::endl;
std::cout << "  Slice domain: 2D (" << slice.domain().size() << " elements)" << std::endl;
std::cout << "  Slice extent<Y>(): " << slice.extent<Y>() << std::endl;
std::cout << "  Slice extent<Z>(): " << slice.extent<Z>() << std::endl;
double dx = 0.1, dy = 0.2, dz = 0.05;
auto grid_dims = data.extents();
double Lx = ddc::select<X>(grid_dims).value() * dx;
double Ly = ddc::select<Y>(grid_dims).value() * dy;
double Lz = ddc::select<Z>(grid_dims).value() * dz;
    
    std::cout << "Physical domain size:" << std::endl;
    std::cout << "  X: " << data.extent<X>() << " points × " << dx 
              << " spacing = " << Lx << " units" << std::endl;
    std::cout << "  Y: " << data.extent<Y>() << " points × " << dy 
              << " spacing = " << Ly << " units" << std::endl;
    std::cout << "  Z: " << data.extent<Z>() << " points × " << dz 
              << " spacing = " << Lz << " units" << std::endl;

  
  
  

  return 0;
}

