#include "TDTransportDDC/TransportEquation.hpp"
#include <stdexcept>
#include <iostream>

namespace td {

TransportEquation::TransportEquation(int nx,int ny,int nz,int G,int Na)
  : nx_(nx), ny_(ny), nz_(nz), G_(G), Na_(Na)
{
  if (nx_<=0||ny_<=0||nz_<=0) throw std::invalid_argument("invalid mesh dims");
  ncells_ = nx_*ny_*nz_;
  phi_.assign(G_*ncells_, 0.0);
  angular_.assign(G_*Na_*ncells_, 0.0);
  velocities_.assign(G_, 1.0);
}

TransportEquation::~TransportEquation() {}

void TransportEquation::set_total_xs(const std::vector<real_t>& sigma_t) { xs_op_.set_total_xs(sigma_t, G_, ncells_); }
void TransportEquation::set_scatter_xs(const std::vector<real_t>& sigma_s) { scat_op_.set_scatter_xs(sigma_s, G_, ncells_); }
void TransportEquation::set_fission_xs(const std::vector<real_t>& nu_sigma_f) { fiss_op_.set_nu_sigma_f(nu_sigma_f, G_, ncells_); }
void TransportEquation::set_chi(const std::vector<real_t>& chi) { fiss_op_.set_chi(chi, G_); }
void TransportEquation::set_external_source(const std::vector<real_t>& qext) { src_op_.set_source(qext, G_, ncells_); }
void TransportEquation::set_velocities(const std::vector<real_t>& velocities) { if ((int)velocities.size()!=G_) throw std::invalid_argument("vel size"); velocities_ = velocities; }

void TransportEquation::step_time(int nsteps, real_t dt) {
  if (nsteps<=0) return;

  std::vector<real_t> temp(G_*ncells_,0.0);
  std::vector<real_t> xs_term(G_*ncells_,0.0);
  std::vector<real_t> scat_term(G_*ncells_,0.0);
  std::vector<real_t> fiss_term(G_*ncells_,0.0);
  std::vector<real_t> src_term(G_*ncells_,0.0);

  for (int step=0; step<nsteps; ++step) {
    // time-derivative handled via backward Euler: (phi^{n+1} - phi^n)/dt
    // Build RHS = source + scattering + fission
    src_op_.apply_host({}, src_term, G_, Na_, ncells_);
    scat_op_.apply_host(phi_, scat_term, G_, Na_, ncells_);
    fiss_op_.apply_host(phi_, fiss_term, G_, Na_, ncells_);

    // total RHS per group: src + scat + fiss
    for (int i=0;i<G_*ncells_;++i) temp[i] = src_term[i] + scat_term[i] + fiss_term[i];

    // Implicit update: phi^{n+1} = (phi^n / dt + RHS) / (1/dt + Sigma_t)
    // Need Sigma_t per group per cell; we can call TotalXSOperator via apply on phi to get Sigma_t*phi,
    // but we require Sigma_t itself; here we reuse a trick: compute denom by evaluating TotalXSOperator on unit vector.
    // For production use store sigma_t separately and use a direct vector.
    std::vector<real_t> ones(G_*ncells_, 1.0);
    std::vector<real_t> sigma_phi(G_*ncells_,0.0);
    xs_op_.apply_host(ones, sigma_phi, G_, Na_, ncells_); // sigma_phi = sigma_t * 1 => sigma_t

    for (int g=0; g<G_; ++g) {
      for (int c=0; c<ncells_; ++c) {
        int idx = g*ncells_ + c;
        real_t denom = 1.0/dt + sigma_phi[idx];
        real_t numer = phi_[idx]/dt + temp[idx];
        phi_[idx] = numer / (denom + 1e-16);
      }
    }
  }
}

std::vector<real_t> TransportEquation::get_phi(int g) const {
  if (g<0||g>=G_) throw std::invalid_argument("group out of range");
  std::vector<real_t> out(ncells_);
  for (int c=0;c<ncells_;++c) out[c] = phi_[g*ncells_ + c];
  return out;
}

} // namespace td
