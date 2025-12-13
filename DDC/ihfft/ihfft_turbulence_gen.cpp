#include <iostream>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
using Excspc = Kokkos::DefaultExecutionSpace;
using Memspc = Excspc::memory_space;
int main(int argc, char* argv[]){
Kokkos::initialize(argc,argv);
{
  int N = 128;
  double L = 1.0;
  Kokkos::View<double*> spectrum_real("S_k",N);
  Kokkos::View<Kokkos::complex<double>* ,Memspc> field_complex("u_x",N/2 + 1);
  Kokkos::parallel_for("InitSpectrum",N , KOKKOS_LAMBDA(const int i){
    double x = (double)i / N * L;
    spectrum_real(i) = std::exp(-100*(x - 0.5) * (x - 0.5));
  });
  Kokkos::fence();
  auto h_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),spectrum_real);
  for (int i = 0; i<std::size(h_host); i++){
    std::cout<<"Spectrum real ("<< i <<" ) = "<<spectrum_real(i)<<" \t";
  }
  std::cout<<std::endl;
  KokkosFFT::ihfft(Excspc(),spectrum_real,field_complex);
  Kokkos::fence();
  auto h_host2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),field_complex);
  for(int i = 0; i<std::size(h_host2); i++){
      std::cout<<"Complex out ("<< i <<" ) = "<<h_host2(i)<<" \t";
  }
std::cout<<std::endl;
}
Kokkos::finalize();
return 0;
}
