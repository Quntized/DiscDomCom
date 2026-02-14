#include <ddc/ddc.hpp>
#include <iostream>

struct X;
struct Y;

struct GridX {
  using continuous_dimension_type = X;
};
struct GridY {
  using continuous_dimension_type = Y;
};
KOKKOS_FUNCTION ddc::Coordinate<X> coordinate(ddc::DiscreteElement<GridX> const& elem){
  return ddc::Coordinate<X>(0.0+elem.uid()*0.1);
};
KOKKOS_FUNCTION ddc::Coordinate<Y> coordinate(ddc::DiscreteElement<GridY> const& elem){
  return ddc::Coordinate<Y>(0.0+elem.uid()*0.1);
};
int main(int argc, char** argv){
  Kokkos::ScopeGuard kokkos_scope( argc,argv);
  ddc::DiscreteElement<GridX,GridY> grid_point(10,20);
  ddc::Coordinate<X,Y> physical_position = ddc::coordinate(grid_point);
  std::cout<<"Grid indices = "<<grid_point.uid<GridX>()<<" , "<<grid_point.uid<GridY>()<<std::endl;
  //std::cout<<"Grid indices = "<<ddc::get<GridX>.uid()<<" , "<<ddc::get<GridY>.uid()<<std::endl;
  std::cout<<"Physical Position : ( "<<ddc::select<X>(physical_position)<<" , "<<ddc::get<Y>(physical_position)<<std::endl;
  return 0;
}
