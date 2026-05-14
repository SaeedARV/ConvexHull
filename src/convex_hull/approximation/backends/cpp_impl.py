from __future__ import annotations

import numpy as np

try:
    import convex_hull_cpp
except Exception as exc:  # pragma: no cover
    convex_hull_cpp = None
    _import_error = exc
else:
    _import_error = None


class CppBackend:
    name = "cpp"

    def argmax_dot(self, points: np.ndarray, directions: np.ndarray) -> np.ndarray:
        if convex_hull_cpp is None:
            raise ImportError(f"convex_hull_cpp not available: {_import_error}")
        return np.asarray(convex_hull_cpp.argmax_dot(points, directions), dtype=int)

    def apply_householder_to_rows(self, base_dirs: np.ndarray, v_house: np.ndarray) -> np.ndarray:
        if convex_hull_cpp is None:
            raise ImportError(f"convex_hull_cpp not available: {_import_error}")
        return np.asarray(convex_hull_cpp.apply_householder_to_rows(base_dirs, v_house), dtype=float)
