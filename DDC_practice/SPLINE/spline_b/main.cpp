#include <ddc/ddc.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ddc/kernels/splines.hpp> 
#include <Kokkos_Core.hpp>
struct X {
  static constexpr bool PERIODIC = false;
};
struct X2 { static constexpr bool PERIODIC = false;};
struct GridX: ddc::UniformPointSampling<X> {};
struct Theta { static constexpr bool PERIODIC = true;};
struct BSplineTheta: ddc::UniformBSplines<Theta, 3> {};
struct BSplineX : ddc::UniformBSplines<X,3> {};
struct BSplineX2 : ddc::NonUniformBSplines<X2,3> {};
using KnotsX = ddc::UniformBsplinesKnots<BSplineX>;
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
  std::array<double, BSplineX::degree() + 1> deriv_ptr;
  Kokkos::mdspan<double,Kokkos::extents<std::size_t,BSplineX::degree() + 1>> kp(deriv_ptr.data());
  //ddc::Coordinate<X> x(0.35);
  auto first_bspline_d = bsplines.eval_deriv(kp,x);
  std::cout<<"First B Spline index: "<<first_bspline_d.uid()<<std::endl;
  double sum2 = 0.0;
  for(std::size_t i =0; i < BSplineX::degree() + 1; ++i){
    std::cout<<" dB/dx [ "<<first_bspline.uid() + i << " ]  =   "<<deriv_ptr[i]<<std::endl;
    sum2 += deriv_ptr[i];
  }
  std::cout<<"\n sum of derivative = "<<sum<<std::endl;
  double h = 1e-06;
  std::array<double , BSplineX::degree() + 1> vp,vm;
  Kokkos::mdspan<double,Kokkos::extents<std::size_t , BSplineX::degree() + 1>> p(vp.data()), m(vm.data());
  auto j1 = bsplines.eval_basis(p, ddc::Coordinate<X>(0.35 - h));
  auto j2 = bsplines.eval_basis(m,ddc::Coordinate<X>(0.35 + h));
  for(std::size_t i = 0; i< BSplineX::degree() + 1; ++i){
    double numerical = (vm[i] - vp[i])/(2*h);
    std::cout<<"  B[ "<<i<<" ] = " <<first_bspline_d.uid()<<" ----->  "<< "(analytical )  = "<<deriv_ptr[i]<<"  . (numerical ) -------> "<<numerical<<std::endl;
  }
  std::size_t const n_derivs = 2;
  std::array<double,(BSplineX::degree() + 1) * (n_derivs + 1)> dd_ptr;
  Kokkos::mdspan<double, Kokkos::extents<std::size_t, BSplineX::degree() + 1, n_derivs + 1>> dd(dd_ptr.data());
  auto first = bsplines.eval_basis_and_n_derivs(dd, x, n_derivs);
  std::cout<< " First B-spline :  "<< first.uid()<<std::endl;
  for (std::size_t i = 0; i < BSplineX::degree() + 1; ++i) {
    std::cout<<" B[ "<<first.uid()<<" ]    :   "<<std::endl;
    std::cout<<"  value      =   "<<DDC_MDSPAN_ACCESS_OP(dd,i,0)<<std::endl;
    std::cout<<"  1st deriv     =   "<<DDC_MDSPAN_ACCESS_OP(dd,i,1)<<std::endl;
    std::cout<<"  2nd deriv     =     "<<DDC_MDSPAN_ACCESS_OP(dd,i,2)<<std::endl;
  }
  std::cout<< " knots has "<<BSplineX::degree() + 2<<std::endl;
  for (auto elem: full_dom){
    auto first_knot = bsplines.get_first_support_knot(elem);
    auto last_knot = bsplines.get_last_support_knot(elem);
    double x_first = ddc::get<X>(ddc::coordinate(first_knot));
    double x_last = ddc::get<X>(ddc::coordinate(last_knot));
    std::cout<<"  B[ "<<elem.uid()<<" ] : "<<" support = [ "<<x_first<<" , "<<x_last<<" ] "<<std::endl;
  }
  std::cout<<std::boolalpha;
  std::cout<<" BSplineX is uniform bsplines_v = "<<ddc::is_uniform_bsplines_v<BSplineX> << std::endl;
  std::cout<<" degree = "<<BSplineX::degree() <<std::endl;
  std::cout<<" is_ uniform = "<<BSplineX::is_uniform()<<std::endl;
  std::cout<<" is periodic = "<<BSplineX::is_periodic()<<std::endl;
  std::cout<<"BSplineX2 is uniform bspline  =  "<<ddc::is_uniform_bsplines_v<BSplineX2> <<std::endl;
  std::cout<<"int is _uniform bspline? "<<ddc::is_uniform_bsplines_v<int> <<std::endl;
  ddc::init_discrete_space<BSplineTheta>(ddc::Coordinate<Theta>(0.0),ddc::Coordinate<Theta>(2 * M_PI), 8);
  auto const& b = ddc::discrete_space<BSplineTheta>();
  std::cout<<" Is Periodic ? "<<BSplineTheta::is_periodic()<<std::endl;
  std::cout<< " Degree ? "<<BSplineTheta::degree() <<std::endl;
  std::cout<<" Cells ? "<<b.ncells()<<std::endl;
  std::cout<<" nbasis ? "<<b.nbasis()<<std::endl;
  std::cout<< " size : "<<b.size()<<std::endl;
  std::array<double,BSplineTheta::degree() + 1> po;
  Kokkos::mdspan<double, Kokkos::extents<std::size_t , BSplineTheta::degree() + 1>> rp(po.data());
  for(double t : {0.0,1.0,2.0,3.0,5.0,6.0}){
    b.eval_basis(rp,ddc::Coordinate<Theta>(t));
    double su = 0.0;
    for (std::size_t i = 0; i <rp.extent(0);  ++i){
      std::cout<<"po[i]  = "<<po[i]<<std::endl;
      su += po[i];
      //std::cout<<"  theta = "<<t<<" , sum = "<<su<<"\n";
    }
    std::cout<<"  theta = "<<t<<" , sum = "<<su<<"\n";
  }
  
  double knot_step = ddc::step<KnotsX>();
  std::cout<<" Knot step : "<<knot_step<<std::endl;
  auto bp_d = bsplines.break_point_domain();
  std::cout<<" break point domain size : "<<bp_d.size()<<std::endl;
  for (auto elem : bp_d){
    double x = ddc::get<X>(ddc::coordinate(elem));
    std::cout<<" Knot[ "<<elem.uid()<<" ]  =  "<< x <<std::endl;
  }
  
  
  return 0;
}

