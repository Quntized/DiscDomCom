#include <iostream>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
using Excspc = Kokkos::DefaultExecutionSpace;
using Memspc = Excspc::memory_space;

int main(int argc, char* argv[]){
  Kokkos::initialize(argc,argv);
  {
    int N = 128;
    double L = 1.0;
    double c = 0.5; //wave speed;
    double t_final = 1.0;
    Kokkos::View<double*,Memspc> u_real("u_real_space",N);
    Kokkos::View<Kokkos::complex<double>*,Memspc> u_hat("u_hat_cmplx",N/2 + 1);
    Kokkos::parallel_for("Init",N,KOKKOS_LAMBDA(const int i){
      double x = (double) i /N * L;
      u_real(i) = std::exp(-100.00*(x - 0.25)*(x - 0.25));
    });
    Kokkos::fence();
    auto u_hostd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u_real);
    Kokkos::deep_copy(u_hostd ,u_real);
    for (int i=0; i<std::size(u_real); i++){
      std::cout<<"u_real("<<i<<" ) = "<<u_hostd(i)<<" ";
    }
    std::cout<<std::endl;
    auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u_real);
    KokkosFFT::fft(Excspc(),u_real,u_hat);
    auto u_host_hat = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),u_hat);
    Kokkos::fence();
    Kokkos::deep_copy(u_host_hat, u_hat);
    for (int i=0; i<std::size(u_host_hat); i++){
      std::cout<<"u_hat("<<i<<" ) = "<<u_host_hat(i)<<" ";
    }
    std::cout<<std::endl;
    Kokkos::parallel_for("Advect",N/2 + 1, KOKKOS_LAMBDA(const int k_idx){
      double k = 2.0 * M_PI * k_idx / L;
      double theta = k * c * t_final;
      Kokkos::complex<double> phase_shift(std::cos(theta),-std::sin(theta));
      u_hat(k_idx) = u_hat(k_idx) * phase_shift;
    });
    KokkosFFT::irfft(Excspc(),u_hat,u_real);
    Kokkos::fence();
    
    Kokkos::deep_copy(u_host,u_real);
    
    for (int i=0; i<std::size(u_host); i++){
      std::cout<<"u_host("<<i<<" ) = "<<u_host(i)<<" ";
    }
    std::cout<<std::endl;
  }
  Kokkos::finalize();
  return 0;
}
