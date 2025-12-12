#include "TDTransportDDC/TimeDerivative.hpp"
#include <stdexcept>

namespace td {

void TimeDerivative::apply_host(const std::vector<real_t>& phi_old, std::vector<real_t>& out, int G, int Na, int ncells) {
  if ((int)phi_old.size() != G*ncells) throw std::invalid_argument("phi_old size mismatch");
  out.assign(G*ncells, 0.0);
  // This operator just produces the time-derivative contribution placeholder; actual coefficient 1/v and dt
  // are applied in the TransportEquation assembly.
  for (int g=0; g<G; ++g) {
    for (int c=0; c<ncells; ++c) {
      out[g*ncells + c] = phi_old[g*ncells + c];
    }
  }
}

} // namespace td
