#include "TDTransportDDC/StreamingOperator.hpp"
#include <stdexcept>
#include <cmath>

namespace td {

void StreamingOperator::apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) {
  if ((int)in.size() != G*Na*ncells) throw std::invalid_argument("in size mismatch for streaming");
  out.assign(G*Na*ncells, 0.0);

  // Very generic implementation: assume uniform mesh with cell-centered finite-volume upwind.
  // For a real DDC-based implementation use DDC field operations and mesh connectivity.
  for (int g=0; g<G; ++g) {
    for (int a=0; a<Na; ++a) {
      for (int c=0; c<ncells; ++c) {
        // placeholder: streaming derivative approximated as zero (user should supply mesh connectivity)
        out[(g*Na + a)*ncells + c] = 0.0; 
      }
    }
  }
}

} // namespace td
