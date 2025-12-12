#include "TDTransportDDC/ScatteringOperator.hpp"
#include <stdexcept>

namespace td {

void ScatteringOperator::set_scatter_xs(const std::vector<real_t>& sigma_s, int G, int ncells) {
  if ((int)sigma_s.size() != G*G*ncells) throw std::invalid_argument("sigma_s size mismatch");
  sigma_s_ = sigma_s;
}

void ScatteringOperator::apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) {
  if ((int)in.size() != G*ncells) throw std::invalid_argument("in size mismatch for scattering");
  if ((int)sigma_s_.size() != G*G*ncells) throw std::runtime_error("sigma_s not set");

  out.assign(G*ncells, 0.0);
  for (int gto=0; gto<G; ++gto) {
    for (int gfrom=0; gfrom<G; ++gfrom) {
      for (int c=0; c<ncells; ++c) {
        real_t s = sigma_s_[(gto*G + gfrom)*ncells + c];
        out[gto*ncells + c] += s * in[gfrom*ncells + c];
      }
    }
  }
}

} // namespace td
