#include <ddc/ddc.hpp>
#include <iostream>

struct X {};
struct Y{};
struct Z{};

int main() {
  ddc::DiscreteDomain<X,Y,Z> dom(
    ddc::DiscreteElement<X,Y,Z>(0,0,0),
    ddc::DiscreteVector<X,Y,Z>(3,3,3)
  );
  ddc::Chunk<double , ddc::DiscreteDomain<X,Y,Z>> data("Data", dom);
  auto full_dom = ddc::get_domain<X,Y,Z>(data);
  std::cout<<"Full domain size : "<<full_dom.size() <<std::endl;
  auto X_dom = ddc::get_domain<X>(data);
  std::cout<< "X domain size = "<<X_dom.size()<<std::endl;
  std::cout<<"X front : "<<ddc::select<X>(X_dom.front()).uid()<<std::endl;
  std::cout<<"X back: "<<ddc::select<X>(X_dom.back()).uid()<<std::endl;
  auto Y_dom = ddc::get_domain<Y>(data);
  std::cout<< "Y domain size = "<<Y_dom.size()<<std::endl;
  std::cout<<"Y front : "<<ddc::select<Y>(Y_dom.front()).uid()<<std::endl;
  std::cout<<"Y back: "<<ddc::select<Y>(Y_dom.back()).uid()<<std::endl;
  auto XY_dom = ddc::get_domain<X,Y>(data);
  std::cout<<"XY domain size = "<<XY_dom.size()<<std::endl;
  auto slice = data[ddc::DiscreteElement<X>(2)];
  auto slice_dom = ddc::get_domain<Y,Z>(slice);
  std::cout<<"slice Domain size = "<<slice_dom.size()<<std::endl;
  auto slice_Y = ddc::get_domain<Y>(slice);
  std::cout<<"Slice_Y domain size = "<<slice_Y.size()<<std::endl;
  auto span = data.span_view();
  auto span_dom = ddc::get_domain<X,Y,Z>(span);
  std::cout<<"Span size = "<<span_dom.size()<<std::endl;
  auto z_domain = ddc::get_domain<Z>(data);
  ddc::for_each(z_domain , [&](auto elem_z){
    auto k = ddc::select<Z>(elem_z).uid();
    std::cout<<"Z = "<<k<<std::endl;
  });
  
  
  
  

  return 0;
}
