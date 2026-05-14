from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ValidatedPoints:
    points: np.ndarray
    original_points: np.ndarray
    unique_indices: np.ndarray
    inverse_indices: np.ndarray


def _as_array(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError("points must be a 2D array")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("points must have shape (n_points, ndim) with n_points>=1, ndim>=1")
    if not np.isfinite(arr).all():
        raise ValueError("points must be finite")
    return arr


def validate_points(points: np.ndarray, *, deduplicate: bool = True) -> ValidatedPoints:
    arr = _as_array(points)
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    original = arr
    if not deduplicate:
        n = arr.shape[0]
        unique_indices = np.arange(n)
        inverse_indices = np.arange(n)
        return ValidatedPoints(arr, original, unique_indices, inverse_indices)

    unique_points, unique_indices, inverse_indices = np.unique(
        arr, axis=0, return_index=True, return_inverse=True
    )
    return ValidatedPoints(
        points=unique_points,
        original_points=original,
        unique_indices=unique_indices,
        inverse_indices=inverse_indices,
    )


def restore_original_indices(indices: np.ndarray, unique_indices: np.ndarray) -> np.ndarray:
    return unique_indices[indices]


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=int)
    if vertices.ndim != 1:
        raise ValueError("vertices must be a 1D index array")
    return vertices
