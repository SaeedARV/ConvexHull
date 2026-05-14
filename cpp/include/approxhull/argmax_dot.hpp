#pragma once
#include <cstddef>
#include <vector>

namespace approxhull {
std::vector<std::size_t> argmax_dot(const double* points,
                                    std::size_t n_points,
                                    std::size_t dim,
                                    const double* directions,
                                    std::size_t n_dirs);
}
