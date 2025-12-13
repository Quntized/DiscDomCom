#include <iostream>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
using Excspc = Kokkos::DefaultExecutionSpace;
using memspc = Excspc::memory_space;
int main(int argc, char* argv[]){
Kokkos::initialize(argc,argv);
{
int N = 128;
double L = 10.0;
Kokkos::View<Kokkos::complex<double>* ,memspc> u_complex("u_in",N/2 + 1);
Kokkos::View<double*,memspc> u_spectrum("u_out",N);
Kokkos::parallel_for("Init",N/2 + 1,KOKKOS_LAMBDA(const int i){
  double x = ((double)i / N * L ) - (L / 2.0);
  double real_p = std::exp(-x*x);
  double imag_p = x*std::exp(-x*x);
  u_complex(i) = Kokkos::complex<double>(real_p,imag_p);
});
Kokkos::fence();
auto u_host_complex_init = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u_complex);
for(int i = 0; i<std::size(u_host_complex_init); i++){
  std::cout<<"u complex  "<<i<<" ="<<u_complex(i)<<"                                             " ;
}
std::cout<<std::endl;
KokkosFFT::hfft(Excspc(),u_complex,u_spectrum);
Kokkos::fence();
auto spec_host_ = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u_spectrum);
for(int i = 0; i<std::size(spec_host_); i++){
  std::cout<<"u real result  "<<i<<" = "<<spec_host_(i)<<"                                       ";
}
std::cout<<std::endl;
}
Kokkos::finalize();
return 0;
}
