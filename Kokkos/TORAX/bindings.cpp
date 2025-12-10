#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "ToraxKernels.hpp"

namespace nb = nanobind;
using namespace nb::literals; // Required for "_a"

void init_kokkos() {
    if (!Kokkos::is_initialized()) Kokkos::initialize();
}

void finalize_kokkos() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
}

NB_MODULE(torax_kokkos, m) {
    m.def("init", &init_kokkos, "Initialize Kokkos Runtime");
    m.def("finalize", &finalize_kokkos, "Finalize Kokkos Runtime");

    nb::class_<TransportSolver>(m, "TransportSolver")
        .def(nb::init<int, double, double>(), "n_grid"_a, "dt"_a, "radius"_a)
        // FIX: Added argument names here
        .def("set_initial_condition", &TransportSolver::set_initial_condition, "T_core"_a, "T_edge"_a)
        .def("set_transport_model", &TransportSolver::set_transport_model, "chi_coeff"_a)
        .def("step", &TransportSolver::step)
        .def("get_temperature", &TransportSolver::get_temperature);
}
