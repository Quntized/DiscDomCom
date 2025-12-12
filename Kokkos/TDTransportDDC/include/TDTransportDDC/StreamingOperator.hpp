#pragma once
#include "Operators.hpp"

namespace td {

// Streaming operator: Omega · ∇ φ(r, E, Omega, t)
// This class implements streaming for host-array angular fluxes arranged as [g][a][cell].
class StreamingOperator : public IOperator {
public:
  StreamingOperator() = default;
  // in: angular flux flattened size G*Na*ncells (g major, then a, then cell)
  // out: same size, result of Omega·∇ acting on angular flux (discretized with finite-difference / upwind)
  void apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) override;
};

} // namespace td
