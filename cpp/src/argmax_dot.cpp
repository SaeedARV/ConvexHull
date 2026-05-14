#include "approxhull/argmax_dot.hpp"
#include <cstdint>
#include <limits>

#ifdef APPROXHULL_OPENMP
#include <omp.h>
#endif

namespace approxhull {
std::vector<std::size_t> argmax_dot(const double* points,
                                    std::size_t n_points,
                                    std::size_t dim,
                                    const double* directions,
                                    std::size_t n_dirs) {
  std::vector<std::size_t> winners(n_dirs, 0);
#ifdef APPROXHULL_OPENMP
#pragma omp parallel for
#endif
  for (std::int64_t j = 0; j < static_cast<std::int64_t>(n_dirs); ++j) {
    const double* dir = directions + j * dim;
    double best = -std::numeric_limits<double>::infinity();
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i < n_points; ++i) {
      const double* pt = points + i * dim;
      double dot = 0.0;
      for (std::size_t k = 0; k < dim; ++k) {
        dot += pt[k] * dir[k];
      }
      if (dot > best) {
        best = dot;
        best_idx = i;
      }
    }
    winners[static_cast<std::size_t>(j)] = best_idx;
  }
  return winners;
}
}
