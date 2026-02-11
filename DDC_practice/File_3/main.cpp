#include <ddc/ddc.hpp>
#include <iostream>

struct X {};
struct Y {};
struct Z {};

int main() {
  ddc::DiscreteDomain<X,Y,Z> dom(
    ddc::DiscreteElement<X,Y,Z>(0,0,0),
    ddc::DiscreteVector<X,Y,Z>(5,4,3)
  );
  ddc::Chunk<double ,ddc::DiscreteDomain<X,Y,Z>> data(dom);
  
  ddc::for_each(dom, [&](auto elem) {
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    data(elem) = i* 1 + j* 1+ 1*k;
  });
  auto elem = ddc::DiscreteElement<X,Y,Z>(2,3,1);
  std::cout<< "data (2,3,1) = "<<data(elem) << std::endl;
  data(elem) = 999.00;
  std::cout<<"After Modification = "<< data(elem)<<std::endl;
  auto slice_x = data[ddc::DiscreteElement<X>(2)];
  auto slice_domain = slice_x.domain();
  ddc::for_each(slice_domain , [&](auto elem){
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    std::cout<<" Y = "<<j <<" ,Z = "<<k<<" : "<<slice_x(elem)<<std::endl;
  });
  auto slice_xy = data[ddc::DiscreteElement<X,Y>(1,2)];
  auto line_domain  = slice_xy.domain();
  ddc::for_each(line_domain,[&](auto elem){
    auto k = ddc::select<Z>(elem).uid();  // discrete_domain.cpp line 98 see;
    std::cout<<" Z : "<<k<<" "<<slice_xy(elem)<<std::endl;
  });
  ddc::DiscreteDomain<X,Y,Z> subdomain(ddc::DiscreteElement<X,Y,Z>(1,1,1), ddc::DiscreteVector<X,Y,Z>(4,3,2));
  auto point = data[subdomain];
  ddc::for_each(subdomain, [&](auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    std::cout<< " ( "<<i<<", "<<j<<" ,"<<k<<" )"<<" : "<<point(elem)<<std::endl;
});
auto vec = ddc::DiscreteVector<X,Y,Z>(2,1,2);
std::cout<<"Access Vector Offsete by (2,1,2) : "<< data(vec)<<std::endl;
auto mutable_slice = data[ddc::DiscreteElement<X>(3)];
ddc::for_each(mutable_slice.domain(),[&](auto elem){
  //auto i = ddc::select<X>(elem).uid();
  auto j = ddc::select<Y>(elem).uid();
  auto k = ddc::select<Z>(elem).uid();
  mutable_slice(elem) = 777.0;
  std::cout<<"(  "<<j<< " , "<<k<<" )"<<" : "<<mutable_slice(elem)<<std::endl;
});
return 0;
}
