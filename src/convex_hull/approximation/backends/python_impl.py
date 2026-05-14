from __future__ import annotations

import numpy as np

from .._support import argmax_dot_blocked
from .._sampling import apply_householder_to_rows


class PythonBackend:
    name = "python"

    def argmax_dot(self, points: np.ndarray, directions: np.ndarray) -> np.ndarray:
        return argmax_dot_blocked(points, directions)

    def apply_householder_to_rows(self, base_dirs: np.ndarray, v_house: np.ndarray) -> np.ndarray:
        return apply_householder_to_rows(base_dirs, v_house)
