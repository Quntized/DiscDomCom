#include <ddc/ddc.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>
struct X{};
struct Y{};
struct Z{};
int main() {
  Kokkos::ScopeGuard kokkos_scope;
  auto a = ddc::DiscreteElement<X,Y,Z>(0,0,0);
  auto b = ddc::DiscreteVector<X,Y,Z>(4,5,6);
  ddc::DiscreteDomain<X,Y,Z> dom(a,b);
  ddc::Chunk<double,ddc::DiscreteDomain<X,Y,Z>> DD("DiscreteDomain", dom);
  ddc::for_each(dom, [&]( auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    DD(elem) = i + j + k;
  });
  auto span = DD.span_view();
  ddc::DiscreteElement<X,Y,Z> elem(1,2,1);
  std::cout<<span(elem)<<std::endl;
  std::cout<<DD(elem)<<std::endl;
  ddc::DiscreteElement<X> elem_x(2) ;
  ddc::DiscreteElement<Y> elem_y(1) ;
  ddc::DiscreteElement<Z> elem_z(3) ;
  std::cout<<span(elem_x,elem_y,elem_z)<<std::endl;
  std::cout<<DD(elem_x,elem_y,elem_z)<<std::endl;
  ddc::DiscreteElement<X, Y> elem_xy(2, 1);
  std::cout<<span(elem_xy, elem_z)<<std::endl;
  ddc::DiscreteVector<X, Y, Z> vec3d(2, 1, 3);
  std::cout<<span(vec3d)<<std::endl;
  ddc::DiscreteVector<X> vec_x(2);
  ddc::DiscreteVector<Y> vec_y(1);
  ddc::DiscreteVector<Z> vec_z(3);
  std::cout<<span(vec_x, vec_y, vec_z)<<std::endl;
  std::cout<<DD(vec3d)<<std::endl;
  std::cout<<DD(vec_x,vec_y,vec_z)<<std::endl;
  ddc::DiscreteVector<X, Y> vec_xy(2, 1);
  std::cout<<span(vec_xy, vec_z)<<std::endl;
  std::cout<< "span(ddc::DiscreteElement<X,Y,Z>(2,1,3)) = "<<span(ddc::DiscreteElement<X,Y,Z>(2,1,3))<<std::endl;
  std::cout<<"dom.front()  = "<<dom.front()<<std::endl;
  std::cout<<"span(ddc::DiscreteVector<X,Y,Z>(2,1,3))  =  "<<span(ddc::DiscreteVector<X,Y,Z>(2,1,3))<<std::endl;
  auto aa = ddc::DiscreteElement<X,Y,Z>(10,20,30);
  auto bb = ddc::DiscreteVector<X,Y,Z>(14,25,36);
  ddc::DiscreteDomain<X,Y,Z> dom_offset(aa,bb);
  ddc::Chunk<double, ddc::DiscreteDomain<X, Y, Z>> DD_offset("ChunkOffset", dom_offset);
    ddc::for_each(dom_offset, [&]( auto elem){
    auto i = ddc::select<X>(elem).uid();
    auto j = ddc::select<Y>(elem).uid();
    auto k = ddc::select<Z>(elem).uid();
    DD_offset(elem) = i + j + k;
  });
  auto span_offset = DD_offset.span_view();
  std::cout<<"dom_offset.front() = "<<dom_offset.front()<<std::endl;
  std::cout<<"span_offset(ddc::DiscreteElement<X,Y,Z>(12,21,33))  =  "<<span_offset(ddc::DiscreteElement<X,Y,Z>(12,21,33))<<std::endl;
  std::cout<<"span_offset(ddc::DiscreteVector<X,Y,Z>(2,1,3)) = "<<span_offset(ddc::DiscreteVector<X,Y,Z>(2,1,3))<<std::endl;
  std::cout<<"span(ddc::DiscreteElement<X,Y,Z>(1,1,1))  =  "<<span(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  span(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = 86.5;
  std::cout<<"span(ddc::DiscreteElement<X,Y,Z>(1,1,1)) = "<<span(ddc::DiscreteElement<X,Y,Z>(1,1,1))<<std::endl;
  span(ddc::DiscreteVector<X,Y,Z>(1,1,1)) = 56.4;
  std::cout<<"span(ddc::DiscreteElement<X,Y,Z>(1,1,1)) =  "<<span(ddc::DiscreteVector<X,Y,Z>(1,1,1))<<std::endl;
  ddc::DiscreteDomain<X,Y,Z> as(ddc::DiscreteElement<X,Y,Z>(0,0,0),ddc::DiscreteVector<X,Y,Z>(1,2,1));
  auto dd = DD[as];
  ddc::for_each(dd.domain(),[&](auto elem){
    std::cout<<"  "<<dd(elem)<<std::endl;
  });
  std::cout<<dd.size()<<std::endl;
  return 0;
}
