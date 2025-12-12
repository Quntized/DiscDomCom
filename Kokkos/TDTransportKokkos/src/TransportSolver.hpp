#pragma once

#include <Kokkos_Core.hpp>
#include <vector>
#include <memory>

namespace td {

class TransportSolver {
public:
  using real_t = double;

  TransportSolver(int nx, int ny, int nz,
                  int groups,
                  int angles);
  ~TransportSolver();

  // Set material cross sections (per group, per cell)
  void set_total_xs(const std::vector<real_t>& sigma_t);
  void set_scatter_xs(const std::vector<real_t>& sigma_s, int Gfrom, int Gto);
  void set_fission_xs(const std::vector<real_t>& nu_sigma_f);
  void set_chi(const std::vector<real_t>& chi);
  void set_velocities(const std::vector<real_t>& v);

  // External fixed source per group per cell
  void set_external_source(const std::vector<real_t>& q_ext);

  // Run transient simulation: number of time steps, dt, inner iterations per step, tolerance
  void run_time_steps(int nsteps, real_t dt, int inner_iters = 50, real_t tol = 1e-6);

  // Retrieve scalar flux for group g as std::vector
  std::vector<real_t> get_scalar_flux(int g) const;

private:
  int nx_, ny_, nz_, ncells_;
  int G_, Na_;

  // Geometry
  real_t dx_, dy_, dz_;

  // Kokkos views for fields: indexed (g,a,cell) for angular flux and (g,cell) for scalar
  Kokkos::View<real_t***> psi_; // [g][a][cell] (we will view as [G][Na][ncells])
  Kokkos::View<real_t**> phi_;   // [g][cell]

  // Material data (host-side & device views)
  Kokkos::View<real_t*>
      sigma_t_; // size G_*ncells_ (flattened as g*ncells_ + c)
  Kokkos::View<real_t*>
      nu_sigma_f_; // size G_*ncells_
  Kokkos::View<real_t*>
      q_ext_; // external source per group per cell
  Kokkos::View<real_t*>
      velocities_; // G

  // Scattering stored dense per (g_to, g_from, cell) flattened as ((gto*G_+gfrom)*ncells_ + c)
  Kokkos::View<real_t*> sigma_s_;

  // Angular quadrature: directions and weights
  std::vector<real_t> mu_, eta_, xi_, w_; // host

  // helpers
  int idx_gcell(int g, int c) const { return g * ncells_ + c; }
  int idx_scat(int gto, int gfrom, int c) const { return ((gto * G_ + gfrom) * ncells_) + c; }
};

} // namespace td
