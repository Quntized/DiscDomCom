// Common includes and forward declarations for operator interfaces
#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include <memory>

namespace td {

using real_t = double;

// Generic interface for operators acting on the angular flux or scalar flux fields.
// For integration with DDC you can specialize using DDC field types (e.g., ddc::Field<T>). 
struct IOperator {
  virtual ~IOperator() = default;
  // Apply operator: out = op(in) (in/out are flattened host arrays sized G*ncells or Na*G*ncells depending)
  virtual void apply_host(const std::vector<real_t>& in, std::vector<real_t>& out, int G, int Na, int ncells) = 0;
};

} // namespace td
