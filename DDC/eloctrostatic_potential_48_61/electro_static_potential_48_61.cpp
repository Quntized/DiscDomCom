#include <iostream>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
using Excspc = Kokkos::DefaultExecutionSpace;
using Memspc = Excspc::memory_space;
int main(int argc,char* argv[]){
Kokkos::initialize(argc,argv);
{
  int N = 64;
  double L = 1.0;
  Kokkos::View<double*,Memspc> rho("Charge_density",N);
  Kokkos::View<double*, Memspc> phi("Potential",N);
  Kokkos::View<Kokkos::complex<double>*,Memspc> rho_hat("rho_hat",N/2 + 1);
  Kokkos::View<Kokkos::complex<double>*,Memspc> phi_hat("phi_hat",N/2 + 1);
  Kokkos::parallel_for("Init_rho" , N,KOKKOS_LAMBDA(const int i){
    double x = (double) i / N*L;
    rho(i) = std::sin(2.0*M_PI*x);
  });
  Kokkos::fence();
  KokkosFFT::rfft(Excspc(),rho,rho_hat);
  Kokkos::parallel_for("Solve k space",N/2+1,KOKKOS_LAMBDA(const int k_idx){
    if (k_idx == 0){
      phi_hat(k_idx) = 0.0;
    }
    else{
      double k = 2.0*M_PI * k_idx / L;
      phi_hat(k_idx) = rho_hat(k_idx) / (k*k);
    }
  });
  Kokkos::fence();
  
  KokkosFFT::irfft(Excspc(),phi_hat,phi);
  Kokkos::fence();
  auto phi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),phi);
  std::cout<<"Result . The size of phi = "<<std::size(phi_hat)<<std::endl;
  for (int i = 0; i<std::size(phi_host); i++){
    std::cout<<phi_host(i)<<" ";
  }
  std::cout<<std::endl;
}
Kokkos::finalize();
return 0;
}
