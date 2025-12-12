#pragma once
#include "Operators.hpp"
#include "TimeDerivative.hpp"
#include "StreamingOperator.hpp"
#include "TotalXSOperator.hpp"
#include "ScatteringOperator.hpp"
#include "FissionOperator.hpp"
#include "ExternalSource.hpp"
#include <vector>

namespace td {

// Compose all operator terms and provide time stepping driver.
class TransportEquation {
public:
  TransportEquation(int nx,int ny,int nz,int G,int Na);
  ~TransportEquation();

  // setters for materials and sources
  void set_total_xs(const std::vector<real_t>& sigma_t);
  void set_scatter_xs(const std::vector<real_t>& sigma_s);
  void set_fission_xs(const std::vector<real_t>& nu_sigma_f);
  void set_chi(const std::vector<real_t>& chi);
  void set_external_source(const std::vector<real_t>& qext);
  void set_velocities(const std::vector<real_t>& velocities);

  // run transient
  void step_time(int nsteps, real_t dt);

  // access scalar flux (g flattened)
  std::vector<real_t> get_phi(int g) const;

private:
  int nx_,ny_,nz_,ncells_,G_,Na_;
  TimeDerivative td_op_;
  StreamingOperator st_op_;
  TotalXSOperator xs_op_;
  ScatteringOperator scat_op_;
  FissionOperator fiss_op_;
  ExternalSource src_op_;

  std::vector<real_t> phi_; // scalar flux G*ncells
  std::vector<real_t> angular_; // angular flux Na*G*ncells
  std::vector<real_t> velocities_; // G
};

} // namespace td
