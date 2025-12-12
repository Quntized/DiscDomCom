#pragma once
#include "Operators.hpp"

namespace td {

// External source term s(r,E,Omega,t) — stored per group per cell as scalar (isotropic) for now
class ExternalSource : public IOperator {
public:
  ExternalSource() = default;
  void set_source(const std::vector<real_t>& q_ext, int G, int ncells);
  void apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) override;
private:
  std::vector<real_t> q_ext_; // G*ncells
};

} // namespace td
