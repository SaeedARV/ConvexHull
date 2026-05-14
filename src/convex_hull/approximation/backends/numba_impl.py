from __future__ import annotations

import numpy as np

try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None

from .._sampling import apply_householder_to_rows


if nb is not None:

    @nb.njit(parallel=True)
    def _argmax_dot_numba(points: np.ndarray, directions: np.ndarray) -> np.ndarray:
        n_dirs = directions.shape[0]
        n_points = points.shape[0]
        dim = points.shape[1]
        winners = np.empty(n_dirs, dtype=np.int64)
        for j in nb.prange(n_dirs):
            best = -1.0e308
            best_idx = 0
            for i in range(n_points):
                dot = 0.0
                for k in range(dim):
                    dot += points[i, k] * directions[j, k]
                if dot > best:
                    best = dot
                    best_idx = i
            winners[j] = best_idx
        return winners


class NumbaBackend:
    name = "numba"

    def argmax_dot(self, points: np.ndarray, directions: np.ndarray) -> np.ndarray:
        if nb is None:
            raise ImportError("numba is not installed")
        return _argmax_dot_numba(points, directions)

    def apply_householder_to_rows(self, base_dirs: np.ndarray, v_house: np.ndarray) -> np.ndarray:
        return apply_householder_to_rows(base_dirs, v_house)
