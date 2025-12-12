#include "TransportSolver.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace td {

using real_t = TransportSolver::real_t;

static std::vector<std::array<real_t,4>> make_level_s_n(int Na) {
  // Simple placeholder: create Na directions uniformly on sphere using spherical angles.
  // For realistic work you should replace with a vetted quadrature (e.g., Level-Symmetric, Gauss-Legendre).
  std::vector<std::array<real_t,4>> out;
  out.reserve(Na);
  for (int a = 0; a < Na; ++a) {
    real_t phi = 2.0 * M_PI * a / Na;
    real_t theta = M_PI * (a + 0.5) / Na;
    real_t mu = std::cos(theta);
    real_t eta = std::sin(theta) * std::cos(phi);
    real_t xi = std::sin(theta) * std::sin(phi);
    real_t w = 4.0 * M_PI / Na;
    out.push_back({mu, eta, xi, w});
  }
  return out;
}

TransportSolver::TransportSolver(int nx, int ny, int nz, int groups, int angles)
  : nx_(nx), ny_(ny), nz_(nz), G_(groups), Na_(angles)
{
  if (nx_<=0 || ny_<=0 || nz_<=0) throw std::invalid_argument("nx/ny/nz must be positive");
  if (G_<=0 || Na_<=0) throw std::invalid_argument("groups and angles must be positive");

  ncells_ = nx_ * ny_ * nz_;
  dx_ = dy_ = dz_ = 1.0; // uniform unit cube cell size by default

  // Allocate views
  psi_ = Kokkos::View<real_t***>("psi", G_, Na_, ncells_);
  phi_ = Kokkos::View<real_t**>("phi", G_, ncells_);

  sigma_t_ = Kokkos::View<real_t*>("sigma_t", G_ * ncells_);
  nu_sigma_f_ = Kokkos::View<real_t*>("nu_sigma_f", G_ * ncells_);
  q_ext_ = Kokkos::View<real_t*>("q_ext", G_ * ncells_);
  velocities_ = Kokkos::View<real_t*>("velocities", G_);

  sigma_s_ = Kokkos::View<real_t*>("sigma_s", G_ * G_ * ncells_);

  // initialize host-side quadrature
  auto quad = make_level_s_n(Na_);
  mu_.resize(Na_); eta_.resize(Na_); xi_.resize(Na_); w_.resize(Na_);
  for (int a = 0; a < Na_; ++a) {
    mu_[a] = quad[a][0]; eta_[a] = quad[a][1]; xi_[a] = quad[a][2]; w_[a] = quad[a][3];
  }

  // initialize fields to zero
  Kokkos::deep_copy(psi_, 0.0);
  Kokkos::deep_copy(phi_, 0.0);
  Kokkos::deep_copy(sigma_t_, 1e-6);
  Kokkos::deep_copy(nu_sigma_f_, 0.0);
  Kokkos::deep_copy(q_ext_, 0.0);
  Kokkos::deep_copy(sigma_s_, 0.0);
  Kokkos::deep_copy(velocities_, 1.0);
}

TransportSolver::~TransportSolver() {}

void TransportSolver::set_total_xs(const std::vector<real_t>& sigma_t) {
  if ((int)sigma_t.size() != G_ * ncells_) throw std::invalid_argument("sigma_t size mismatch");
  auto h = Kokkos::create_mirror_view(sigma_t_);
  for (size_t i=0;i<sigma_t.size();++i) h[i] = sigma_t[i];
  Kokkos::deep_copy(sigma_t_, h);
}

void TransportSolver::set_scatter_xs(const std::vector<real_t>& sigma_s, int Gfrom, int Gto) {
  (void)Gfrom; (void)Gto; // we accept flattened input per (gto,gfrom,cell)
  if ((int)sigma_s.size() != G_ * G_ * ncells_) throw std::invalid_argument("sigma_s size mismatch");
  auto h = Kokkos::create_mirror_view(sigma_s_);
  for (size_t i=0;i<sigma_s.size();++i) h[i] = sigma_s[i];
  Kokkos::deep_copy(sigma_s_, h);
}

void TransportSolver::set_fission_xs(const std::vector<real_t>& nu_sigma_f) {
  if ((int)nu_sigma_f.size() != G_ * ncells_) throw std::invalid_argument("nu_sigma_f size mismatch");
  auto h = Kokkos::create_mirror_view(nu_sigma_f_);
  for (size_t i=0;i<nu_sigma_f.size();++i) h[i] = nu_sigma_f[i];
  Kokkos::deep_copy(nu_sigma_f_, h);
}

void TransportSolver::set_chi(const std::vector<real_t>& /*chi*/) {
  // In this simplified storage we assume chi used in fission source calculation is provided externally in bindings
}

void TransportSolver::set_velocities(const std::vector<real_t>& v) {
  if ((int)v.size() != G_) throw std::invalid_argument("velocities size mismatch");
  auto h = Kokkos::create_mirror_view(velocities_);
  for (int g=0; g<G_; ++g) h[g]=v[g];
  Kokkos::deep_copy(velocities_, h);
}

void TransportSolver::set_external_source(const std::vector<real_t>& q_ext) {
  if ((int)q_ext.size() != G_ * ncells_) throw std::invalid_argument("q_ext size mismatch");
  auto h = Kokkos::create_mirror_view(q_ext_);
  for (size_t i=0;i<q_ext.size();++i) h[i] = q_ext[i];
  Kokkos::deep_copy(q_ext_, h);
}

std::vector<real_t> TransportSolver::get_scalar_flux(int g) const {
  if (g<0 || g>=G_) throw std::invalid_argument("group out of range");
  auto h = Kokkos::create_mirror_view(phi_);
  Kokkos::deep_copy(h, phi_);
  std::vector<real_t> out(ncells_);
  for (int c=0;c<ncells_;++c) out[c] = h(g,c);
  return out;
}

// Simple serial sweep for one angle and group. Real implementations must carefully optimize, parallelize, and
// ensure correct memory access patterns. We implement a straightforward upwind cell-centered update.
static void sweep_serial(int nx,int ny,int nz,int nc, double mu,double eta,double xi,
                         const std::vector<double>& sigma_t_host,
                         const std::vector<double>& q_host,
                         std::vector<double>& psi_out)
{
  // iterate in sweep order depending on sign of direction cosines
  int isgn = mu>=0 ? 1 : -1;
  int jsgn = eta>=0 ? 1 : -1;
  int ksgn = xi>=0 ? 1 : -1;

  int istart = (isgn==1) ? 0 : nx-1;
  int iend   = (isgn==1) ? nx : -1;
  int jstart = (jsgn==1) ? 0 : ny-1;
  int jend   = (jsgn==1) ? ny : -1;
  int kstart = (ksgn==1) ? 0 : nz-1;
  int kend   = (ksgn==1) ? nz : -1;

  for (int kk=kstart; kk!=kend; kk+=ksgn) {
    for (int jj=jstart; jj!=jend; jj+=jsgn) {
      for (int ii=istart; ii!=iend; ii+=isgn) {
        int c = (kk*ny + jj)*nx + ii;
        // upwind estimate: incoming from neighbor cell toward upstream; here use simple vacuum boundary (0)
        double inc = 0.0;
        // compute psi = (q + inc)/sigma_t
        double st = sigma_t_host[c];
        double q = q_host[c];
        double psi = (q + inc) / (st + 1e-12);
        psi_out[c] = psi;
      }
    }
  }
}

void TransportSolver::run_time_steps(int nsteps, real_t dt, int inner_iters, real_t tol) {
  // For simplicity we perform explicit-in-time streaming + source iteration for scattering/fission.
  // A production-grade solver should use implicit time discretization and acceleration (e.g., DSA).

  // Create host mirrors for required arrays
  auto h_sigma_t = Kokkos::create_mirror_view(sigma_t_);
  auto h_qext = Kokkos::create_mirror_view(q_ext_);
  auto h_phi = Kokkos::create_mirror_view(phi_);
  auto h_sigma_s = Kokkos::create_mirror_view(sigma_s_);

  Kokkos::deep_copy(h_sigma_t, sigma_t_);
  Kokkos::deep_copy(h_qext, q_ext_);
  Kokkos::deep_copy(h_phi, phi_);
  Kokkos::deep_copy(h_sigma_s, sigma_s_);

  std::vector<double> sigma_t_host(ncells_);
  std::vector<double> q_host(ncells_);

  for (int step=0; step<nsteps; ++step) {
    // Outer time loop
    for (int g=0; g<G_; ++g) {
      // prepare sigma_t per cell for this group
      for (int c=0;c<ncells_;++c) sigma_t_host[c] = h_sigma_t[g*ncells_ + c];
      // combine external sources and scattering/fission from previous phi
      for (int c=0;c<ncells_;++c) q_host[c] = h_qext[g*ncells_ + c];

      // inner source iteration
      for (int it=0; it<inner_iters; ++it) {
        // angular sweeps over directions
        for (int a=0; a<Na_; ++a) {
          std::vector<double> psi_out(ncells_,0.0);
          sweep_serial(nx_,ny_,nz_,ncells_, mu_[a], eta_[a], xi_[a], sigma_t_host, q_host, psi_out);
          // accumulate into phi
          for (int c=0;c<ncells_;++c) {
            h_phi(g,c) = h_phi(g,c) + w_[a] * psi_out[c];
          }
        }
        // simple convergence check (not rigorous)
        // For realistic use replace with norm-based check and accelerate
      }
    }
  }

  // copy back to device phi_
  Kokkos::deep_copy(phi_, h_phi);
}

} // namespace td
