#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

namespace py = pybind11;

// Ensure Kokkos is initialized exactly once per process; do not finalize
// immediately so repeated Python calls work
static bool g_kokkos_initialized = false;
static void ensure_kokkos_initialized() {
  if (!g_kokkos_initialized) {
    int dummy_argc = 0;
    Kokkos::initialize(dummy_argc, nullptr);
    g_kokkos_initialized = true;
  }
}

// 1D diffusion explicit solver using finite differences
// Returns final solution as numpy array (shape: [Nx])
py::array_t<double> simulate_diffusion(
    double diffusivity, // D
    double length,      // L
    int Nx,             // number of grid points
    double dt,          // time step
    double t_final,     // final time
    const std::string& bc = "dirichlet", // 'dirichlet' or 'periodic'
    int save_every = 0  // if >0, save snapshots every this many steps (not returned by default)
) {
  ensure_kokkos_initialized();

  if (Nx < 3) throw std::runtime_error("Nx must be >= 3");
  if (dt <= 0) throw std::runtime_error("dt must be > 0");
  if (t_final < 0) throw std::runtime_error("t_final must be >= 0");

  const double dx = length / (Nx - 1);
  const int nsteps = static_cast<int>(std::ceil(t_final / dt));
  const double alpha = diffusivity * dt / (dx * dx);

  if (alpha > 0.5 && bc != "periodic") {
    // warn user about stability for explicit scheme (Dirichlet/Neumann)
    std::cerr << "Warning: explicit scheme may be unstable (alpha=" << alpha
              << "). Consider reducing dt or using implicit scheme.\n";
  }

  Kokkos::View<double*> u_old("u_old", Nx);
  Kokkos::View<double*> u_new("u_new", Nx);

  // initialize: Gaussian centered in the domain with width = L/10
  const double x0 = 0.5 * length;
  const double sigma = 0.1 * length;

  Kokkos::parallel_for("init",
                       Kokkos::RangePolicy<>(0, Nx),
                       KOKKOS_LAMBDA(const int i) {
                         const double x = i * dx;
                         u_old(i) = std::exp(-0.5 * ((x - x0) / sigma) * ((x - x0) / sigma));
                       });
  Kokkos::fence();

  auto interior_update_dirichlet = KOKKOS_LAMBDA(const int i) {
    // i from 1 to Nx-2
    u_new(i) = u_old(i) + alpha * (u_old(i + 1) - 2.0 * u_old(i) + u_old(i - 1));
  };

  auto interior_update_periodic = KOKKOS_LAMBDA(const int i) {
    int left = (i == 0) ? (Nx - 1) : (i - 1);
    int right = (i == Nx - 1) ? 0 : (i + 1);
    u_new(i) = u_old(i) + alpha * (u_old(right) - 2.0 * u_old(i) + u_old(left));
  };

  if (nsteps <= 0) {
    // nothing to do; return initial condition
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_old);
    py::array_t<double> out({Nx});
    auto buf = out.mutable_unchecked<1>();
    for (int i = 0; i < Nx; ++i) buf(i) = h(i);
    return out;
  }

  for (int step = 0; step < nsteps; ++step) {
    if (bc == "dirichlet") {
      // enforce Dirichlet BCs: u(0)=u(Nx-1)=0
      Kokkos::deep_copy(Kokkos::subview(u_new, 0), 0.0);
      Kokkos::deep_copy(Kokkos::subview(u_new, Nx - 1), 0.0);

      // interior
      Kokkos::parallel_for("update",
                           Kokkos::RangePolicy<>(1, Nx - 1),
                           interior_update_dirichlet);

    } else if (bc == "periodic") {
      // periodic: update all indices using periodic neighbors
      Kokkos::parallel_for("update_periodic",
                           Kokkos::RangePolicy<>(0, Nx),
                           interior_update_periodic);
    } else {
      throw std::runtime_error("Unsupported boundary condition: " + bc);
    }

    Kokkos::fence();

    // swap views
    std::swap(u_old, u_new);
  }

  // copy result to host and return numpy array
  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_old);
  py::array_t<double> out({Nx});
  auto buf = out.mutable_unchecked<1>();
  for (int i = 0; i < Nx; ++i) buf(i) = h_out(i);

  return out;
}

// Simple CLI wrapper for demonstration as an executable
int cli_main(int argc, char* argv[]) {
  // parse simple args: Nx dt t_final D
  if (argc < 5) {
    std::cout << "Usage: diffusion_kokkos_cli D length Nx dt t_final [bc]\n";
    std::cout << "Example: diffusion_kokkos_cli 0.1 1.0 201 1e-5 0.1 dirichlet\n";
    return 1;
  }
  double D = std::stod(argv[1]);
  double L = std::stod(argv[2]);
  int Nx = std::stoi(argv[3]);
  double dt = std::stod(argv[4]);
  double t_final = std::stod(argv[5]);
  std::string bc = "dirichlet";
  if (argc > 6) bc = argv[6];

  ensure_kokkos_initialized();
  auto result = simulate_diffusion(D, L, Nx, dt, t_final, bc);
  auto r = result.unchecked<1>();
  for (int i = 0; i < Nx; ++i) std::cout << r(i) << (i + 1 == Nx ? '\n' : ' ');

  return 0;
}

PYBIND11_MODULE(diffusion_kokkos, m) {
  m.doc() = "1D diffusion solver implemented with Kokkos and exposed via pybind11.";
  m.def("simulate",
        &simulate_diffusion,
        py::arg("diffusivity"),
        py::arg("length"),
        py::arg("Nx"),
        py::arg("dt"),
        py::arg("t_final"),
        py::arg("bc") = "dirichlet",
        py::arg("save_every") = 0,
        "Run a 1D explicit diffusion simulation and return final field as numpy array.");
}
