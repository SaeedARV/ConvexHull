"""Compatibility wrapper for uniform directional approximate hulls."""

from __future__ import annotations

from typing import Optional

import numpy as np

from convex_hull.approximation import ApproxConvexHull
from convex_hull.approximation._random import get_rng
from convex_hull.approximation._sampling import sample_uniform_sphere
from convex_hull.approximation.backends import get_backend
from convex_hull.utils import normalize


def _backend_name(method: str) -> str:
    method = method.lower()
    if method == "numpy":
        return "python"
    if method == "gpu":
        return "auto"
    return method


def _deterministic_directions(t: int, d: int) -> np.ndarray:
    if d == 2:
        theta = np.linspace(0.0, 2.0 * np.pi, int(t), endpoint=False)
        return np.stack((np.cos(theta), np.sin(theta)), axis=1)
    rng = np.random.default_rng(0)
    return sample_uniform_sphere(int(t), d, rng)


class ConvexHullviaExtents:
    """Uniform directional approximate hull solver.

    New code should use `ApproxConvexHull(..., method="uniform")` directly.
    This class preserves the old experiment-facing method names.
    """

    def __init__(self, points, *, random_state: Optional[object] = None):
        self.points = np.ascontiguousarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError("points must be a 2D array")
        self.X = self.points
        self.random_state = random_state
        self.last_hull: Optional[ApproxConvexHull] = None

    def approximate(
        self,
        t: int = 1000,
        *,
        method: str = "auto",
        random_state: Optional[object] = None,
        include_ties: bool = False,
    ) -> ApproxConvexHull:
        self.last_hull = ApproxConvexHull(
            self.points,
            method="uniform",
            backend=_backend_name(method),
            m=int(t),
            random_state=self.random_state if random_state is None else random_state,
            include_ties=include_ties,
        )
        return self.last_hull

    def compute_indices(
        self,
        t: int = 1000,
        *,
        mode: str = "random",
        method: str = "auto",
        random_state: Optional[object] = None,
    ) -> np.ndarray:
        if mode.lower() != "random":
            directions = _deterministic_directions(t, self.points.shape[1])
            winners = get_backend(_backend_name(method)).argmax_dot(self.points, directions)
            return np.unique(winners.astype(int))
        return self.approximate(t, method=method, random_state=random_state).vertices

    def _extents_for_directions(self, directions: np.ndarray, method: str):
        winners = get_backend(_backend_name(method)).argmax_dot(self.points, directions)
        extents = {tuple(point): [] for point in self.points}
        for idx, direction in zip(winners, directions):
            extents[tuple(self.points[int(idx)])].append(direction)
        return extents, winners

    def get_extents(self, t=1000, return_max_extent=False, method: str = "auto"):
        directions = _deterministic_directions(int(t), self.points.shape[1])
        extents, _ = self._extents_for_directions(directions, method)
        if not return_max_extent:
            return extents

        dot_products = self.points @ directions.T
        max_idx_per_point = np.argmax(dot_products, axis=1)
        extents_max = {
            tuple(point): normalize(directions[int(direction_idx)])
            for point, direction_idx in zip(self.points, max_idx_per_point)
        }
        return extents, extents_max

    def get_random_extents(
        self,
        t,
        return_approx_hull=False,
        method: str = "auto",
        random_state: Optional[object] = None,
    ):
        rng = get_rng(self.random_state if random_state is None else random_state)
        directions = sample_uniform_sphere(int(t), self.points.shape[1], rng)
        random_extents, winners = self._extents_for_directions(directions, method)
        if not return_approx_hull:
            return random_extents
        approx_indices = np.unique(winners.astype(int))
        approx_hull = {tuple(self.points[int(idx)]) for idx in approx_indices}
        return random_extents, approx_hull

    def compute(
        self,
        t=1000,
        mode: str = "deterministic",
        method: str = "auto",
        return_max_extent: bool = False,
        return_approx_hull: bool = False,
    ):
        mode = mode.lower()
        if mode == "deterministic":
            return self.get_extents(t=t, return_max_extent=return_max_extent, method=method)
        if mode == "random":
            return self.get_random_extents(t=t, return_approx_hull=return_approx_hull, method=method)
        raise ValueError("mode must be 'deterministic' or 'random'.")
