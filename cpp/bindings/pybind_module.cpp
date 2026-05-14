#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include "approxhull/argmax_dot.hpp"
#include "approxhull/householder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(convex_hull_cpp, m) {
  m.doc() = "C++ kernels for convex_hull approximation backends";

  m.def("argmax_dot", [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
                           py::array_t<double, py::array::c_style | py::array::forcecast> directions) {
    auto pbuf = points.request();
    auto dbuf = directions.request();
    if (pbuf.ndim != 2 || dbuf.ndim != 2) {
      throw std::runtime_error("points and directions must be 2D arrays");
    }
    const auto n_points = static_cast<std::size_t>(pbuf.shape[0]);
    const auto dim = static_cast<std::size_t>(pbuf.shape[1]);
    const auto n_dirs = static_cast<std::size_t>(dbuf.shape[0]);
    if (static_cast<std::size_t>(dbuf.shape[1]) != dim) {
      throw std::runtime_error("dimension mismatch between points and directions");
    }
    const double* pptr = static_cast<double*>(pbuf.ptr);
    const double* dptr = static_cast<double*>(dbuf.ptr);
    auto winners = approxhull::argmax_dot(pptr, n_points, dim, dptr, n_dirs);
    py::array_t<std::size_t> out(static_cast<py::ssize_t>(n_dirs));
    auto obuf = out.request();
    auto* optr = static_cast<std::size_t*>(obuf.ptr);
    for (std::size_t i = 0; i < n_dirs; ++i) {
      optr[i] = winners[i];
    }
    return out;
  });

  m.def("apply_householder_to_rows", [](py::array_t<double, py::array::c_style | py::array::forcecast> base_dirs,
                                          py::array_t<double, py::array::c_style | py::array::forcecast> v_house) {
    auto bbuf = base_dirs.request();
    auto vbuf = v_house.request();
    if (bbuf.ndim != 2 || vbuf.ndim != 1) {
      throw std::runtime_error("base_dirs must be 2D and v_house must be 1D");
    }
    const auto n_dirs = static_cast<std::size_t>(bbuf.shape[0]);
    const auto dim = static_cast<std::size_t>(bbuf.shape[1]);
    if (static_cast<std::size_t>(vbuf.shape[0]) != dim) {
      throw std::runtime_error("dimension mismatch for v_house");
    }
    const double* bptr = static_cast<double*>(bbuf.ptr);
    const double* vptr = static_cast<double*>(vbuf.ptr);
    auto out_vec = approxhull::apply_householder_to_rows(bptr, n_dirs, dim, vptr);
    py::array_t<double> out({
        static_cast<py::ssize_t>(n_dirs),
        static_cast<py::ssize_t>(dim),
    });
    auto obuf = out.request();
    auto* optr = static_cast<double*>(obuf.ptr);
    std::copy(out_vec.begin(), out_vec.end(), optr);
    return out;
  });
}
