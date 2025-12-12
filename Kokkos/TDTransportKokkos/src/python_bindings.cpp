#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Kokkos_Core.hpp>
#include "TransportSolver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tdtransport, m) {
  if (!Kokkos::is_initialized()) {
    int argc = 0; char** argv = nullptr;
    Kokkos::initialize(argc, argv);
  }

  py::class_<td::TransportSolver>(m, "TransportSolver")
    .def(py::init<int,int,int,int,int>(),
         py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("groups"), py::arg("angles"))
    .def("set_total_xs", &td::TransportSolver::set_total_xs)
    .def("set_scatter_xs", &td::TransportSolver::set_scatter_xs)
    .def("set_fission_xs", &td::TransportSolver::set_fission_xs)
    .def("set_velocities", &td::TransportSolver::set_velocities)
    .def("set_external_source", &td::TransportSolver::set_external_source)
    .def("run_time_steps", &td::TransportSolver::run_time_steps,
         py::arg("nsteps"), py::arg("dt"), py::arg("inner_iters") = 50, py::arg("tol") = 1e-6)
    .def("get_scalar_flux", &td::TransportSolver::get_scalar_flux);

  m.doc() = "Time-dependent neutron transport solver (Kokkos + pybind11)";
}
