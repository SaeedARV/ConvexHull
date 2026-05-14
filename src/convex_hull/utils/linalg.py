"""Linear algebra helpers used by convex hull solvers."""

import numpy as np


def normalize(v):
    """Normalize a vector to unit length."""
    return v / np.linalg.norm(v)


def householder_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the Householder reflection H such that H @ a = b.

    Parameters are assumed to be unit vectors in R^d.
    """
    v = a - b
    v = v / np.linalg.norm(v)
    return np.eye(a.shape[0]) - 2.0 * np.outer(v, v)
