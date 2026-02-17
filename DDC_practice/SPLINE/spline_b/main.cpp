#include <ddc/ddc.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ddc/kernels/splines.hpp> 
#include <Kokkos_Core.hpp>
struct X {
  static constexpr bool PERIODIC = false;
};
struct GridX: ddc::UniformPointSampling<X> {};

struct BSplineX : ddc::UniformBSplines<X,3> {};


int main(int argc, char** argv){
  ddc::ScopeGuard ddc_scope(argc,argv);
  std::cout<<std::fixed<<std::setprecision(6);
  std::cout<<" "<<std::endl;
  double x_min = 0.0;
  double x_max = 1.0;
  std::size_t ncells = 10;

  ddc::init_discrete_space<BSplineX>(ddc::Coordinate<X>(0.0),ddc::Coordinate<X>(1.0),ncells);
  auto const bsplines = ddc::discrete_space<BSplineX>();
  std::cout<<" Degree : "<<BSplineX::degree() <<std::endl;
  std::cout<<" Cells: "<<bsplines.ncells()<<std::endl;
  std::cout<<" nbasis: "<<bsplines.nbasis()<<std::endl;
  std::cout<<" rmin: "<<bsplines.rmin()<<std::endl;
  std::cout<<" rmax: "<<bsplines.rmax()<<std::endl;
  auto full_dom = bsplines.full_domain();
  std::cout<<" Bsplines full domain or size: "<<full_dom.size()<<std::endl;
  std::cout<<"Bsplines full domain  front with uid value: "<<full_dom.front().uid()<<std::endl;
  std::cout<<"Bsplines full domain  front without  uid value: "<<full_dom.front()<<std::endl;
  std::cout<<"Bsplines full domain  back with uid value: "<<full_dom.back().uid()<<std::endl;
  for(auto elem:full_dom){
    std::cout<<" Bspline[ "<<elem.uid()<<" ]"<<std::endl;
  }
  auto bp_dom = bsplines.break_point_domain();
  std::cout<<"Size:  "<<bp_dom.size()<<"\n"<<std::endl;
  std::cout<<"Bsplines break_point_domain size -->  "<<bp_dom.size()<<std::endl;
  std::cout<<"Bsplines break point domain front with uid() ---> "<<bp_dom.front().uid()<<std::endl;
  std::cout<<"Bsplines break_point_domain back with uid() ---> "<<bp_dom.back().uid()<<std::endl;
  for (auto elem: bp_dom ){
    std::cout<<" knot[ "<<elem<<" ]"<< " = "<<ddc::coordinate(elem)<<std::endl;
  }
  std::array<double , BSplineX::degree() + 1>  vals_ptr;
  Kokkos::mdspan<double, Kokkos::extents<std::size_t , BSplineX::degree() + 1>> values(vals_ptr.data());
  ddc::Coordinate<X> x(0.35);
  std::cout<<x<<std::endl;
  auto first_bspline = bsplines.eval_basis(values,x);
  std::cout<<"First non-zero B-spline index = "<<first_bspline.uid()<<std::endl;
  double sum = 0.0;
  for (std::size_t i = 0; i <BSplineX::degree() + 1; ++i){
    std::cout<<"    B [ "<<first_bspline.uid() + i <<" ] "<< vals_ptr[i]<<std::endl; 
    sum += vals_ptr[i];
  }
  for (double xval : {0.0,0.2,0.5,0.8,1.0}){
    ddc::Coordinate<X> xc(xval);
    bsplines.eval_basis(values,xc);
    double s = 0.0;
    for (auto v : vals_ptr) {
      std::cout<<"x = "<<xc<<"  :   "<<v<<std::endl;
      s += v;
    }
    std::cout<<" x= "<<xval<<"sum =  "<<s<<std::endl;
    
  }
  
  return 0;
}

