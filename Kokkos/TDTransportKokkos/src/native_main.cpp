#include <Kokkos_Core.hpp>
#include "TransportSolver.hpp"
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    // small native smoke test
    td::TransportSolver solver(10,10,1,2,8);
    std::vector<double> sigma_t(2*10*10*1, 1.0);
    solver.set_total_xs(sigma_t);
    std::vector<double> qext(2*10*10*1, 1.0);
    solver.set_external_source(qext);
    solver.run_time_steps(1, 0.1);
    auto phi0 = solver.get_scalar_flux(0);
    std::cout << "phi[0] cell0=" << phi0[0] << "\n";
  }
  Kokkos::finalize();
  return 0;
}
