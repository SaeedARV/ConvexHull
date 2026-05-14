"""Minimum-volume enclosing ellipsoid utilities."""

from __future__ import annotations

import numpy as np

from convex_hull.approximation._mvee import Ellipsoid as _ComputedEllipsoid
from convex_hull.approximation._mvee import khachiyan_mvee


class Ellipsoid:
    """Compatibility wrapper around the Khachiyan MVEE implementation.

    The public attributes `center` and `shape_matrix` match the old class.
    The ellipsoid is represented as `(x - center)^T shape_matrix (x - center) <= 1`.
    """

    def __init__(self, points, center=None, shape_matrix=None, method="Khachiyan", tol=1e-7, max_iter=1000):
        self.points = np.ascontiguousarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError("points must be a 2D array")
        self.n, self.d = self.points.shape

        if center is None or shape_matrix is None:
            if method not in {"Khachiyan", "khachiyan"}:
                raise ValueError("Only the Khachiyan MVEE method is supported.")
            computed = khachiyan_mvee(self.points, tol=tol, max_iter=max_iter)
            self.center = computed.center
            self.shape_matrix = computed.A
        else:
            self.center = np.asarray(center, dtype=float)
            self.shape_matrix = np.asarray(shape_matrix, dtype=float)

    @property
    def A(self) -> np.ndarray:
        return self.shape_matrix

    def _computed(self) -> _ComputedEllipsoid:
        return _ComputedEllipsoid(center=self.center, A=self.shape_matrix)

    def project_to_boundary(self, point: np.ndarray) -> np.ndarray:
        return self._computed().project_to_boundary(np.asarray(point, dtype=float))

    def project(self, point: np.ndarray, *_, **__) -> np.ndarray:
        return self.project_to_boundary(point)

    def normal_vector(self, boundary_point: np.ndarray) -> np.ndarray:
        return self._computed().normal_vector(np.asarray(boundary_point, dtype=float))

    def contains(self, point: np.ndarray, *, tol: float = 1e-8) -> bool:
        return self._computed().contains(np.asarray(point, dtype=float), tol=tol)

    def compute(self, method="Khachiyan"):
        if method not in {"Khachiyan", "khachiyan"}:
            raise ValueError("Only the Khachiyan MVEE method is supported.")
        computed = khachiyan_mvee(self.points)
        return computed.center, computed.A
