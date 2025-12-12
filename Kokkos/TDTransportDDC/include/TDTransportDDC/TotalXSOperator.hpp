#pragma once
#include "Operators.hpp"

namespace td {

// Total cross-section operator: Sigma_t(r,E) * phi
class TotalXSOperator : public IOperator {
public:
  TotalXSOperator() = default;
  // sigma_t: flattened by G*ncells
  void set_total_xs(const std::vector<real_t>& sigma_t, int G, int ncells);
  void apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) override;
private:
  std::vector<real_t> sigma_t_; // size G*ncells
};

} // namespace td
