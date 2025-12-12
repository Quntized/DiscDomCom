#pragma once
#include "Operators.hpp"

namespace td {

// Scattering operator: integral over E' and Omega' of Sigma_s(r, E' -> E, Omega'->Omega) phi(r,E',Omega')
// We store scattering as a dense matrix per cell: S[gto][gfrom][cell]
class ScatteringOperator : public IOperator {
public:
  ScatteringOperator() = default;
  void set_scatter_xs(const std::vector<real_t>& sigma_s, int G, int ncells);
  // in: scalar flux per (g, cell) flattened, out: same
  void apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) override;
private:
  std::vector<real_t> sigma_s_; // size G*G*ncells flattened as (gto*G + gfrom)*ncells + c
};

} // namespace td
