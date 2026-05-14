"""Shared utilities."""

from convex_hull.utils.data import sample_input
from convex_hull.utils.linalg import householder_matrix, householder_vector, normalize

__all__ = ["householder_matrix", "householder_vector", "normalize", "sample_input"]
