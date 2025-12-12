#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "TDTransportDDC/TransportEquation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tdtransport_ddc, m) {
  m.doc() = "TD transport equation operators and solver with optional DDC integration";

  py::class_<td::TransportEquation>(m, "TransportEquation")
    .def(py::init<int,int,int,int,int>(),
         py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("groups"), py::arg("angles"))
    .def("set_total_xs", &td::TransportEquation::set_total_xs)
    .def("set_scatter_xs", &td::TransportEquation::set_scatter_xs)
    .def("set_fission_xs", &td::TransportEquation::set_fission_xs)
    .def("set_chi", &td::TransportEquation::set_chi)
    .def("set_external_source", &td::TransportEquation::set_external_source)
    .def("set_velocities", &td::TransportEquation::set_velocities)
    .def("step_time", &td::TransportEquation::step_time)
    .def("get_phi", &td::TransportEquation::get_phi);

}
