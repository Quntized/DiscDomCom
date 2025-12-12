#include "TDTransportDDC/ExternalSource.hpp"
#include <stdexcept>

namespace td {

void ExternalSource::set_source(const std::vector<real_t>& q_ext, int G, int ncells) {
  if ((int)q_ext.size() != G*ncells) throw std::invalid_argument("q_ext size mismatch");
  q_ext_ = q_ext;
}

void ExternalSource::apply_host(const std::vector<real_t>& /*in*/, std::vector<real_t>& out, int G, int Na, int ncells) {
  if ((int)q_ext_.size() != G*ncells) throw std::runtime_error("q_ext not set");
  out = q_ext_;
}

} // namespace td
