#include "TDTransportDDC/TransportEquation.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    td::TransportEquation solver(10,10,1,2,8);
    int nx=10, ny=10, nz=1;
    int ncells = nx*ny*nz;
    std::vector<double> sigma_t(2*ncells, 1.0);
    std::vector<double> qext(2*ncells, 0.1);
    solver.set_total_xs(sigma_t);
    solver.set_external_source(qext);
    solver.step_time(1, 0.01);
    auto phi0 = solver.get_phi(0);
    std::cout << "phi0[0]=" << phi0[0] << "\n";
  }
  Kokkos::finalize();
  return 0;
}
