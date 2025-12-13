#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstddef>
#include <ddc/aligned_allocator.hpp>
constexpr std::size_t ALIGNMENT = 64;
using AlignV = std::vector<double , ddc::AlignedAllocator<double,ALIGNMENT>>;
void func(const AlignV& B_old,AlignV& B_new , double eta,double dt, double dx){
  int N = B_old.size();
  double alpha = (eta*dt)/(dx*dx);
  const double* __restrict__ p_old = B_old.data();
  double* __restrict__ p_new = B_new.data();
  #pragma omp simd aligned(p_old,p_new : ALIGNMENT)
  for (int i =1; i<N - 1; ++i){
    p_new[i] = p_old[i] + alpha * (p_old[i+1] - 2.0*p_old[i] + p_old[i-1]);
  }
}
int main(){
  int Nx = 20000000;
  double L = 1.0;
  double eta = 0.001;
  double t_final = 0.01;
  double dx = L/(Nx - 1);
  double dt = 0.4*dx*dx / eta; //stability limit;
  std::cout<<"Allocating "<<(Nx*sizeof(double) / 1024 / 1024)<<"MB of memory  \n";
  AlignV B_curr(Nx);
  AlignV B_next(Nx);
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(B_curr.data());
  std::cout<< "Memory address : "<<B_curr.data()<<"\n";
  if (addr% ALIGNMENT == 0){ 
    std::cout<<"[Success] memory is perfectly aligned to "<<ALIGNMENT<<"bytes. \n";
  }
  else {
    std::cout<<"[Fail] memory is unaligned!\n"; return 1;
  }
  #pragma omp parallel for 
  for (int i = 0; i < Nx; ++i){
    double x = i*dx;
    B_curr[i] = std::exp(-100.0 * (x - 0.5)*(x - 0.5));
  }
  B_curr[0] = 0.0; B_curr[Nx - 1] = 0.0;
  B_next[0] = 0.0; B_next[Nx - 1] = 0.0;
  std::cout<<"Starting simulation \n";
  auto start = std::chrono::high_resolution_clock::now();
  int steps = 0;
  double time = 0.0;
  while (time<t_final){
    func(B_curr,B_next,eta,dt,dx);
    std::swap(B_curr,B_next);
    time += dt;
    steps++;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout<<"Simulation complete .\n";
  std::cout<<"Steps taken "<<steps<<"\n";
  std::cout<<"Time elapsed : "<<diff.count()<<" s.\n";
  std::cout<<"Centered B field value : "<<B_curr[Nx/2]<< "(Should be less than 1 due to diffusion)\n";
  return 0;
  
}
