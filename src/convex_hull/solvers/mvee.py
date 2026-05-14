"""Compatibility wrapper for MVEE-guided vMF approximate hulls."""

from __future__ import annotations

from typing import Optional

import numpy as np

from convex_hull.approximation import ApproxConvexHull
from convex_hull.approximation._random import get_rng
from convex_hull.approximation._sampling import sample_vmf_base


def _backend_name(method: str) -> str:
    method = method.lower()
    if method == "numpy":
        return "python"
    if method == "gpu":
        return "auto"
    return method


class ConvexHullviaMVEE:
    """MVEE-guided vMF approximate hull solver.

    New code should use `ApproxConvexHull(..., method="mvee_vmf")` directly.
    This class preserves the old experiment-facing API.
    """

    def __init__(
        self,
        points,
        *,
        backend: str = "auto",
        random_state: Optional[object] = None,
        mvee_tol: float = 1e-7,
        mvee_max_iter: int = 1000,
    ):
        self.points = np.ascontiguousarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError("points must be a 2D array")
        self.backend = backend
        self.random_state = random_state
        self.mvee_tol = mvee_tol
        self.mvee_max_iter = mvee_max_iter
        self.last_hull: Optional[ApproxConvexHull] = None

    def approximate(
        self,
        m: int = 3,
        kappa: float = 5,
        *,
        backend: Optional[str] = None,
        random_state: Optional[object] = None,
        include_ties: bool = False,
    ) -> ApproxConvexHull:
        self.last_hull = ApproxConvexHull(
            self.points,
            method="mvee_vmf",
            backend=_backend_name(self.backend if backend is None else backend),
            m=int(m),
            kappa=float(kappa),
            random_state=self.random_state if random_state is None else random_state,
            include_ties=include_ties,
            mvee_tol=self.mvee_tol,
            mvee_max_iter=self.mvee_max_iter,
        )
        return self.last_hull

    def compute_indices(
        self,
        m: int = 3,
        kappa: float = 5,
        *,
        backend: Optional[str] = None,
        random_state: Optional[object] = None,
    ) -> np.ndarray:
        return self.approximate(m=m, kappa=kappa, backend=backend, random_state=random_state).vertices

    def extents_estimation(self, U, E, return_extents=True):
        # Retained only for old visualization/debug code. The production path is `compute`.
        hull = self.approximate(m=len(U), kappa=1.0)
        selected = self.points[hull.vertices]
        if not return_extents:
            return selected
        empty_by_point = {tuple(point): [] for point in self.points}
        return selected, empty_by_point, {}, {}

    def compute(self, m=3, kappa=5, return_extents=False, **kwargs):
        hull = self.approximate(m=m, kappa=kappa, backend=kwargs.pop("backend", None), **kwargs)
        selected = self.points[hull.vertices]
        if not return_extents:
            return selected

        rng = get_rng(self.random_state)
        base_dirs = sample_vmf_base(int(m), self.points.shape[1], float(kappa), rng)
        empty_by_point = {tuple(point): [] for point in self.points}
        return selected, empty_by_point, base_dirs, {}, {}
