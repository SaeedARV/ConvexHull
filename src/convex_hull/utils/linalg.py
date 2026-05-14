"""Linear algebra helpers used by convex hull solvers."""

import numpy as np

from convex_hull.approximation._sampling import householder_matrix, householder_vector


def normalize(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return np.asarray(v, dtype=float)
    return np.asarray(v, dtype=float) / norm
