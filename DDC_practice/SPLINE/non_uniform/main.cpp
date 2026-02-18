#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
struct X {static constexpr bool PERIODIC = false;};
static constexpr std::size_t D = 3;
struct BSplineX:ddc::NonUniformBSplines<X,D> {};
void print_seperator(const char* title){
  std::cout<<"\n"<<title<<"\n"<<std::endl;
}
int main(int argc , char** argv) {
  Kokkos::ScopeGuard const kokkos_scope(argc,argv);
  ddc::ScopeGuard const ddc_scope(argc,argv);
  
  std::vector<ddc::Coordinate<X>> const breaks = {ddc::Coordinate<X>(0.0),ddc::Coordinate<X>(0.15),ddc::Coordinate<X>(0.4),ddc::Coordinate<X>(0.6),ddc::Coordinate<X>(0.85),ddc::Coordinate<X>(1.0)};
  ddc::init_discrete_space<BSplineX>(breaks);
  auto const& bsplines = ddc::discrete_space<BSplineX>();
  auto bp_dom = bsplines.break_point_domain();
  for (auto knot:bp_dom){
    std::cout<<ddc::get<X>(ddc::coordinate(knot))<<" "<<std::endl;
  }
  std::cout<<"Domain [ "<<bsplines.rmin()<<" , "<<bsplines.rmax()<<" ] "<<" . BSplines degree --> "<<BSplineX::degree()<<". ncells  --> "<<bsplines.ncells()<<" . nbasis() ---> "<<bsplines.nbasis()<<std::endl;
  std::cout<<"Cell Width Non uniform"<<std::endl;
  for(std::size_t i = 0; i < breaks.size()-1; ++i ){
    double h = ddc::get<X>(breaks[i+1]) - ddc::get<X>(breaks[i]);
    std::cout<<"   Cell "<< i << " : [ "<<ddc::get<X>(breaks[i])<<" , "<<ddc::get<X>(breaks[i+1])<<" ] , width = "<<h<<"\n";
  }
  auto n_basis = bsplines.nbasis();
  std::vector<double> c(n_basis);
    c[0] = 0.0;
    c[1] = 0.5;
    c[2] = 1.2;
    c[3] = 1.8;
    c[4] = 1.5;
    c[5] = 0.8;
    c[6] = 0.3;
    c[7] = 0.0;
    for (std::size_t i = 0; i < n_basis; ++i) {
      std::cout << c[i];
      if (i < n_basis - 1) std::cout << ", ";
    }
   ddc::Coordinate<X> const x_eval(0.37);
   {
        std::array<double, D + 1> vals {};
        ddc::DSpan1D values(vals.data(), D + 1);
        ddc::DiscreteElement<BSplineX> const jmin = bsplines.eval_basis(values, x_eval);
        std::cout << "  eval_basis(x = " << ddc::get<X>(x_eval) << "):\n\n";
        std::cout << "  Returned jmin = " << jmin.uid() << "\n\n";
        std::cout << "  Returned jmin = " << jmin << "\n\n";
        
   }
  return 0;
}
