#pragma once
#include <cstddef>
#include <vector>

namespace approxhull {
std::vector<double> apply_householder_to_rows(const double* base_dirs,
                                              std::size_t n_dirs,
                                              std::size_t dim,
                                              const double* v_house);
}
