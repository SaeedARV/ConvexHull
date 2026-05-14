#include "approxhull/householder.hpp"
#include <cmath>

namespace approxhull {
std::vector<double> apply_householder_to_rows(const double* base_dirs,
                                              std::size_t n_dirs,
                                              std::size_t dim,
                                              const double* v_house) {
  std::vector<double> out(n_dirs * dim, 0.0);
  for (std::size_t i = 0; i < n_dirs; ++i) {
    const double* row = base_dirs + i * dim;
    double dot = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
      dot += v_house[k] * row[k];
    }
    for (std::size_t k = 0; k < dim; ++k) {
      out[i * dim + k] = row[k] - 2.0 * v_house[k] * dot;
    }
  }
  return out;
}
}
