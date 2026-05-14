from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Ellipsoid:
    center: np.ndarray
    A: np.ndarray

    def project_to_boundary(self, point: np.ndarray) -> np.ndarray:
        diff = point - self.center
        denom = np.sqrt(diff @ self.A @ diff)
        if denom == 0.0:
            return self.center.copy()
        return self.center + diff / denom

    def normal_vector(self, boundary_point: np.ndarray) -> np.ndarray:
        vec = self.A @ (boundary_point - self.center)
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def contains(self, point: np.ndarray, *, tol: float = 1e-8) -> bool:
        diff = point - self.center
        return diff @ self.A @ diff <= 1.0 + tol


def khachiyan_mvee(points: np.ndarray, tol: float = 1e-7, max_iter: int = 1000) -> Ellipsoid:
    points = np.ascontiguousarray(points, dtype=float)
    n, d = points.shape
    q = np.vstack([points.T, np.ones(n)])
    u = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        x = q @ np.diag(u) @ q.T
        try:
            x_inv = np.linalg.inv(x)
        except np.linalg.LinAlgError:
            reg = 1e-12 * np.eye(d + 1)
            x_inv = np.linalg.inv(x + reg)
        m = np.einsum("ij,jk,ki->i", q.T, x_inv, q)
        max_idx = np.argmax(m)
        max_val = m[max_idx]
        step = (max_val - d - 1.0) / ((d + 1.0) * (max_val - 1.0))
        new_u = (1.0 - step) * u
        new_u[max_idx] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u

    center = points.T @ u
    cov = points.T @ np.diag(u) @ points - np.outer(center, center)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    A = (1.0 / d) * cov_inv
    vals = np.einsum("ij,jk,ik->i", points - center, A, points - center)
    max_val = np.max(vals)
    if max_val > 1.0:
        A = A / max_val
    return Ellipsoid(center=center, A=A)
