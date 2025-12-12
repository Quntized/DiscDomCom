#pragma once
#include "Operators.hpp"

namespace td {

// Time derivative term: (1/v) * d_phi/dt. We implement a backward-Euler discretization helper
// that computes contribution to the RHS and handles phi^{n+1} update implicitly on scalar flux.
class TimeDerivative : public IOperator {
public:
  TimeDerivative() = default;
  // For host arrays: phi_old and phi_new are flattened by group*ncells
  void apply_host(const std::vector<real_t>& phi_old, std::vector<real_t>& out, int G, int Na, int ncells) override;
  // compute update for implicit step: solves (1/dt * 1/v + Sigma_t) * phi^{n+1} = RHS
};

} // namespace td
