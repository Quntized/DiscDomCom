#pragma once
#include "Operators.hpp"

namespace td {

// Fission operator: chi(E) * nu * Sigma_f(E') * integral over E' of phi(E')
// For multigroup: we store nu_sigma_f per group and chi per group.
class FissionOperator : public IOperator {
public:
  FissionOperator() = default;
  void set_nu_sigma_f(const std::vector<real_t>& nu_sigma_f, int G, int ncells);
  void set_chi(const std::vector<real_t>& chi, int G);
  void apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) override;
private:
  std::vector<real_t> nu_sigma_f_; // G*ncells
  std::vector<real_t> chi_; // G
};

} // namespace td
