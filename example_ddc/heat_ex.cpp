#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include <ddc/ddc.hpp>
//#include <ddc/kernels/fft.hpp>
#include <Kokkos_Core.hpp>

struct X;
struct DDimX: ddc::UniformPointSampling<X>
{};
struct Y;
struct DDimY: ddc::UniformPointSampling<Y>
{};

struct T;
struct DDimT : ddc::UniformPointSampling<T>
{};

int main(int argc,char** argv){
  Kokkos::ScopeGuard const kokkos_scope(argc, argv);
  ddc::ScopeGuard const ddc_scope(argc, argv);
  double const x_start = 10.0;
  double const x_end = 100.0;
  std::size_t const nxpoints = 10;
  double const y_start = 10;
  double const y_end = 100;
  std::size_t const nypoints = 10;
  auto const domainx = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(ddc::Coordinate<X>(x_start),ddc::Coordinate<X>(x_end),ddc::DiscreteVector<DDimX>(nxpoints)));

   auto const domainy = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(ddc::Coordinate<Y>(y_start),ddc::Coordinate<Y>(y_end),ddc::DiscreteVector<DDimY>(nypoints)));

   //auto& [grid_impl1 , domainx] = result1;
   //auto& [grid_impl2, domainy] = result2;
  
   std::cout<<"Discretization in X and Y"<<std::endl;
   std::cout<<"No of of points each = "<<domainx.size()<<" , "<<domainy.size()<<std::endl;
   std::cout<< "and " << domainy.size()<<std::endl;
   for(ddc::DiscreteElement<DDimX> index:domainx){
    ddc::Coordinate<X> coord1 = ddc::coordinate(index);
    std::cout<< "Index = "<< index.uid() << " . Coordinate X "<<" = "<<coord1 <<std::endl;
   }
   for(ddc::DiscreteElement<DDimY> index:domainy){

    ddc::Coordinate<Y> coord2 = ddc::coordinate(index);
    std::cout<< "Index = "<< index.uid() << " . Coordinate Y"<<" = "<<coord2 <<std::endl;
   }
  return 0;
 
}
