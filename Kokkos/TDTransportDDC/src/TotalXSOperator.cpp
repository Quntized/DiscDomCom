#include "TDTransportDDC/TotalXSOperator.hpp"
#include <stdexcept>

namespace td {

void TotalXSOperator::set_total_xs(const std::vector<real_t>& sigma_t, int G, int ncells) {
  if ((int)sigma_t.size() != G*ncells) throw std::invalid_argument("sigma_t size mismatch");
  sigma_t_ = sigma_t;
}

void TotalXSOperator::apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) {
  // in is scalar flux per group per cell (size G*ncells) OR could be angular flux if caller flattens differently.
  // Here we assume caller passes scalar flux (G*ncells). To support angular, caller must assemble accordingly.
  if ((int)in.size() != G*ncells) throw std::invalid_argument("in size mismatch for total xs");
  if ((int)sigma_t_.size() != G*ncells) throw std::runtime_error("sigma_t not set");

  out.assign(G*ncells, 0.0);
  for (int g=0; g<G; ++g) {
    for (int c=0; c<ncells; ++c) {
      out[g*ncells + c] = sigma_t_[g*ncells + c] * in[g*ncells + c];
    }
  }
}

} // namespace td
