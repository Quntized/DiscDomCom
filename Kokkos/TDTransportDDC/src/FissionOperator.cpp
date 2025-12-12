#include "TDTransportDDC/FissionOperator.hpp"
#include <stdexcept>

namespace td {

void FissionOperator::set_nu_sigma_f(const std::vector<real_t>& nu_sigma_f, int G, int ncells) {
  if ((int)nu_sigma_f.size() != G*ncells) throw std::invalid_argument("nu_sigma_f size mismatch");
  nu_sigma_f_ = nu_sigma_f;
}

void FissionOperator::set_chi(const std::vector<real_t>& chi, int G) {
  if ((int)chi.size() != G) throw std::invalid_argument("chi size mismatch");
  chi_ = chi;
}

void FissionOperator::apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) {
  if ((int)in.size() != G*ncells) throw std::invalid_argument("in size mismatch for fission");
  if ((int)nu_sigma_f_.size() != G*ncells) throw std::runtime_error("nu_sigma_f not set");
  if ((int)chi_.size() != G) throw std::runtime_error("chi not set");

  // compute emission: for each target group gto, sum over gfrom nu_sigma_f(gfrom)*phi(gfrom)
  std::vector<real_t> fission_source(G*ncells, 0.0);
  for (int gfrom=0; gfrom<G; ++gfrom) {
    for (int c=0; c<ncells; ++c) {
      fission_source[gfrom*ncells + c] = nu_sigma_f_[gfrom*ncells + c] * in[gfrom*ncells + c];
    }
  }

  out.assign(G*ncells, 0.0);
  for (int gto=0; gto<G; ++gto) {
    for (int c=0; c<ncells; ++c) {
      // chi_gto * sum_{gfrom} nu_sigma_f(gfrom)*phi(gfrom)
      real_t sum = 0.0;
      for (int gfrom=0; gfrom<G; ++gfrom) sum += fission_source[gfrom*ncells + c];
      out[gto*ncells + c] = chi_[gto] * sum;
    }
  }
}

} // namespace td
