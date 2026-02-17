#include <iostream>
#include <cmath>
#include <ddc/ddc.hpp>
#include "ddc_aliases.hpp"

struct X {
  static constexpr bool PERIODIC = false;
}
struct GridX : UniformGridBase<X> {};

int main(int argc, char** argv){
  Kokkos::ScopeGuard kokkos_scope(argc,argv);
  ddc::Scopeguard ddc_scope(argc,argv);
  constexpr double x_min = 0.0;
  constexpr double x_max = 1.0;
  constexpr std::size_t num_points = 101;
  ddc::init_discrete_space<GridX>(GridX::init<GridX>(Coord<X>(x_min),Coord<X>(x_max),IdxStep<GridX>(num_points)));

}
